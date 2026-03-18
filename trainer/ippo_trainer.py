# trainer/ippo_trainer.py
"""
IPPO trainer (independent actors + independent critics).

Usage:
    python -m trainer.ippo_trainer

(Or run directly: python trainer/ippo_trainer.py)

Notes:
- Expects envs.FactoryEnv available and configs/default.yaml
- Each agent has its own critic on local observations (no central critic)
"""
import os
import time
import argparse
import yaml
import numpy as np
from collections import defaultdict, deque

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from envs.factory_env import FactoryEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mlp(input_dim, hidden_sizes=(128, 128), output_dim=None, activation=nn.Tanh):
    layers = []
    last = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last, h))
        layers.append(activation())
        last = h
    if output_dim is not None:
        layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class ActorNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(128, 128)):
        super().__init__()
        self.net = mlp(obs_dim, hidden, None, activation=nn.ReLU)
        self.logits = nn.Linear(hidden[-1], act_dim)

    def forward(self, obs):
        x = self.net(obs)
        return self.logits(x)


class CriticNet(nn.Module):
    def __init__(self, input_dim, hidden=(128, 128)):
        super().__init__()
        self.net = mlp(input_dim, hidden, 1, activation=nn.ReLU)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_gae_np(rewards, values, dones, last_value, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        mask = 0.0 if dones[t] else 1.0
        next_val = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae
    returns = adv + np.array(values, dtype=np.float32)
    return adv, returns


class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.logps = []
        self.rewards = []
        self.dones = []
        self.values = []

    def append(self, obs, action, logp, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.logps.append(logp)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


class IPPOTrainer:
    def __init__(self, env, config):
        self.env = env
        self.agent_ids = env.agent_ids

        tcfg = config.get("training", {})
        self.total_timesteps = int(tcfg.get("total_timesteps", 200000))
        self.n_steps = int(tcfg.get("n_steps", 2048))
        self.batch_size = int(tcfg.get("batch_size", 64))
        self.lr = float(tcfg.get("learning_rate", 3e-4))
        self.gamma = float(tcfg.get("gamma", 0.99))
        self.lam = float(tcfg.get("gae_lambda", 0.95))
        self.clip = float(tcfg.get("clip_range", 0.2))
        self.epochs = int(tcfg.get("update_epochs", 10))
        self.ent_coef = float(tcfg.get("ent_coef", 0.01))
        self.vf_coef = float(tcfg.get("vf_coef", 1.0))
        self.max_grad_norm = float(tcfg.get("max_grad_norm", 0.5))

        # build networks
        self.actors = {}
        self.critics = {}  # per-agent critics
        self.optim_actor = None
        self.optim_critic = None  # dict per agent

        self._build_networks()

        # buffers per agent
        self.buffers = {a: RolloutBuffer() for a in self.agent_ids}

        # logging
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=f"logs/ippo_{stamp}")
        self.global_step = 0
        self.episode_count = 0

    def _build_networks(self):
        obs_dims = {a: self.env.get_observation_space(a).shape[0] for a in self.agent_ids}
        act_dims = {a: self.env.get_action_space(a).n for a in self.agent_ids}

        # actors (independent)
        for a in self.agent_ids:
            net = ActorNet(obs_dims[a], act_dims[a]).to(DEVICE)
            self.actors[a] = net

        # IPPO: per-agent critics
        self.critics = {}
        self.optim_critic = {}
        for a in self.agent_ids:
            c = CriticNet(obs_dims[a]).to(DEVICE)
            self.critics[a] = c
            self.optim_critic[a] = torch.optim.Adam(c.parameters(), lr=self.lr)
        self.optim_actor = torch.optim.Adam([p for a in self.agent_ids for p in self.actors[a].parameters()], lr=self.lr)

    def _select_actions_and_values(self, obs_dict):
        """
        Return:
         - action_dict (int chosen)
         - logp_dict (float)
         - value_dict: per-agent value
        """
        action_dict = {}
        logp_dict = {}
        value_dict = {}

        # IPPO: per-agent critic
        for a in self.agent_ids:
            obs_t = torch.tensor(obs_dict[a].astype(np.float32), dtype=torch.float32, device=DEVICE).unsqueeze(0)
            # actor
            logits = self.actors[a](obs_t)
            dist = Categorical(logits=logits)
            act_t = dist.sample()
            act = act_t.detach().cpu().numpy().item()
            logp = dist.log_prob(act_t).detach().cpu().numpy().item()
            # value from local critic
            with torch.no_grad():
                v = self.critics[a](obs_t).cpu().numpy().item()
            action_dict[a] = int(act)
            logp_dict[a] = float(logp)
            value_dict[a] = float(v)
        return action_dict, logp_dict, value_dict

    def train(self):
        obs_dict = self.env.reset()
        ep_rewards = deque(maxlen=100)
        ep_reward_acc = defaultdict(float)

        while self.global_step < self.total_timesteps:
            # collect rollouts
            for step in range(self.n_steps):
                action_dict, logp_dict, value_dict = self._select_actions_and_values(obs_dict)
                next_obs, reward_dict, done_dict, info_dict = self.env.step(action_dict)

                # store per agent
                for a in self.agent_ids:
                    self.buffers[a].append(
                        obs=obs_dict[a].copy(),
                        action=action_dict[a],
                        logp=logp_dict[a],
                        reward=reward_dict.get(a, 0.0),
                        done=done_dict.get(a, False),
                        value=value_dict[a],
                    )
                    ep_reward_acc[a] += reward_dict.get(a, 0.0)

                obs_dict = next_obs
                self.global_step += 1

                if done_dict.get("__all__", False):
                    total = sum(ep_reward_acc.values())
                    ep_rewards.append(total)
                    self.writer.add_scalar("episode/total_reward", total, self.episode_count)
                    print(f"[Episode {self.episode_count}] total_reward={total:.2f} global_step={self.global_step}")
                    ep_reward_acc = defaultdict(float)
                    self.episode_count += 1
                    obs_dict = self.env.reset()

                if self.global_step >= self.total_timesteps:
                    break

            # compute last values for bootstrap (per-agent)
            last_value = {}
            for a in self.agent_ids:
                obs_t = torch.tensor(obs_dict[a].astype(np.float32), dtype=torch.float32, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    last_value[a] = self.critics[a](obs_t).cpu().numpy().item()

            # Build dataset (aggregate samples across agents)
            dataset = []
            for a in self.agent_ids:
                buf = self.buffers[a]
                if len(buf) == 0:
                    continue
                # compute advantages & returns
                advs, rets = compute_gae_np(buf.rewards, buf.values, buf.dones, last_value[a], self.gamma, self.lam)
                for i in range(len(buf)):
                    dataset.append({
                        "agent": a,
                        "obs": buf.obs[i],
                        "action": buf.actions[i],
                        "old_logp": buf.logps[i],
                        "return": rets[i],
                        "adv": advs[i],
                    })
                buf.clear()

            if len(dataset) == 0:
                continue

            # normalize advantages
            advs_all = np.array([d["adv"] for d in dataset], dtype=np.float32)
            adv_mean, adv_std = advs_all.mean(), advs_all.std() + 1e-8
            for d in dataset:
                d["adv"] = (d["adv"] - adv_mean) / adv_std

            # training loop
            dataset_size = len(dataset)
            indices = np.arange(dataset_size)
            minibatch = max(1, min(self.batch_size, dataset_size // 4))

            for epoch in range(self.epochs):
                np.random.shuffle(indices)
                for start in range(0, dataset_size, minibatch):
                    batch_idx = indices[start:start+minibatch]
                    # accumulate losses
                    total_actor_loss = 0.0
                    total_critic_loss = 0.0
                    total_entropy = 0.0
                    for idx in batch_idx:
                        s = dataset[idx]
                        a_id = s["agent"]
                        obs = torch.tensor(s["obs"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        act = torch.tensor(s["action"], dtype=torch.long, device=DEVICE).unsqueeze(0)
                        old_logp = torch.tensor(s["old_logp"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        ret = torch.tensor(s["return"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        adv = torch.tensor(s["adv"], dtype=torch.float32, device=DEVICE).unsqueeze(0)

                        # actor loss
                        logits = self.actors[a_id](obs)
                        dist = Categorical(logits=logits)
                        new_logp = dist.log_prob(act.squeeze(-1)).unsqueeze(0)
                        entropy = dist.entropy().mean()
                        ratio = torch.exp(new_logp - old_logp)
                        surr1 = ratio * adv
                        surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
                        a_loss = -torch.min(surr1, surr2).mean()

                        # critic loss
                        vpred = self.critics[a_id](obs).unsqueeze(0)
                        c_loss = (ret - vpred).pow(2).mean()

                        total_actor_loss += a_loss
                        total_critic_loss += c_loss
                        total_entropy += entropy

                    # normalize
                    mb = float(len(batch_idx))
                    total_actor_loss = total_actor_loss / mb
                    total_critic_loss = total_critic_loss / mb
                    total_entropy = total_entropy / mb

                    # actor step
                    self.optim_actor.zero_grad()
                    (total_actor_loss - self.ent_coef * total_entropy).backward()
                    nn.utils.clip_grad_norm_([p for a in self.agent_ids for p in self.actors[a].parameters()], self.max_grad_norm)
                    self.optim_actor.step()

                    # critic step(s): per-agent
                    # We step each optimizer using only samples for that agent.
                    for a in self.agent_ids:
                        agent_indices = [idx for idx in batch_idx if dataset[idx]["agent"] == a]
                        if len(agent_indices) == 0:
                            continue
                        c_loss_agent = 0.0
                        for idx in agent_indices:
                            s = dataset[idx]
                            obs = torch.tensor(s["obs"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                            ret = torch.tensor(s["return"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                            vpred = self.critics[a](obs).unsqueeze(0)
                            c_loss_agent += (ret - vpred).pow(2).mean()
                        c_loss_agent = c_loss_agent / float(len(agent_indices))
                        self.optim_critic[a].zero_grad()
                        (self.vf_coef * c_loss_agent).backward()
                        nn.utils.clip_grad_norm_(self.critics[a].parameters(), self.max_grad_norm)
                        self.optim_critic[a].step()

            # logging
            if len(ep_rewards) > 0:
                self.writer.add_scalar("episode/avg_last_100", float(np.mean(ep_rewards)), self.global_step)
                print(f"[global_step {self.global_step}] avg_last_100={float(np.mean(ep_rewards)):.2f}")

        # finish
        self.writer.close()


def load_cfg(path="configs/default.yaml"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    env = FactoryEnv(config_path=args.config)
    trainer = IPPOTrainer(env, cfg)
    trainer.train()
