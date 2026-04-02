import os
import time
import yaml
import math
import random
from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

# local import (ensure project root in PYTHONPATH or run from project root)
from envs.factory_env import FactoryEnv

# ---------------------------
# Utilities / Networks
# ---------------------------
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
        # final policy head
        self.logits = nn.Linear(hidden[-1], act_dim)

    def forward(self, obs):
        x = self.net(obs)
        logits = self.logits(x)
        return logits


class CriticNet(nn.Module):
    def __init__(self, central_obs_dim, hidden=(256, 256)):
        super().__init__()
        self.net = mlp(central_obs_dim, hidden, 1, activation=nn.ReLU)

    def forward(self, central_obs):
        return self.net(central_obs).squeeze(-1)  # (batch,)


# ---------------------------
# Rollout Buffer (per-agent storage)
# ---------------------------
class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.central_obs = []  # for critic

    def append(self, obs, action, logp, reward, done, value, central_obs):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(logp)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.central_obs.append(central_obs)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


# ---------------------------
# Helper functions
# ---------------------------
def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """
    rewards, values: list of scalars (len T)
    dones: list of bools
    last_value: scalar bootstrap value for time T
    returns: advantages (np array), returns (np array)
    """
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


# ---------------------------
# Trainer class
# ---------------------------
class MAPPOTrainer:
    def __init__(self, env, config):
        self.env = env
        self.agent_ids = env.agent_ids  # ["mover_1","mover_2","manuf_1"]

        # read training hyperparams (with fallbacks)
        tcfg = config.get("training", {})
        self.total_timesteps = int(tcfg.get("total_timesteps", 200000))
        self.n_steps = int(tcfg.get("n_steps", 2048))
        self.batch_size = int(tcfg.get("batch_size", 64))
        self.learning_rate = float(tcfg.get("learning_rate", 3e-4))
        self.gamma = float(tcfg.get("gamma", 0.99))
        self.gae_lambda = float(tcfg.get("gae_lambda", 0.95))
        self.clip_range = float(tcfg.get("clip_range", 0.2))
        self.update_epochs = int(tcfg.get("update_epochs", 10))
        self.ent_coef = float(tcfg.get("ent_coef", 0.01))
        # Optional epsilon-greedy exploration on top of policy sampling.
        # Useful for sparse reward / many invalid-action environments.
        self.exploration_eps = float(tcfg.get("exploration_eps", 0.0))
        self.use_action_masks = bool(tcfg.get("use_action_masks", True))
        self.vf_coef = float(tcfg.get("vf_coef", 1.0))
        self.max_grad_norm = float(tcfg.get("max_grad_norm", 0.5))
        self.save_freq = int(tcfg.get("save_freq", 50000))

        # per-agent actor nets + optimizer params
        self.actors = {}
        self.optim_actor = None
        self.critic = None
        self.optim_critic = None

        # build networks
        self._build_networks(config)

        # rollout buffers per agent
        self.buffers = {a: RolloutBuffer() for a in self.agent_ids}

        # logging
        self.writer = SummaryWriter(log_dir="logs/ppo_mappo_" + time.strftime("%Y%m%d_%H%M%S"))
        self.global_step = 0
        self.episode_count = 0

    def _build_networks(self, config):
        # get obs dim per agent from env
        obs_dims = {a: self.env.get_observation_space(a).shape[0] for a in self.agent_ids}
        act_dims = {a: self.env.get_action_space(a).n for a in self.agent_ids}

        # central obs dim: concat of all agent obs
        central_obs_dim = sum(obs_dims.values())

        # create actor nets (independent)
        for a in self.agent_ids:
            net = ActorNet(obs_dims[a], act_dims[a]).to(DEVICE)
            self.actors[a] = net

        # centralized critic
        self.critic = CriticNet(central_obs_dim).to(DEVICE)

        # single optimizer for all actors' parameters (you can also separate)
        actor_params = []
        for a in self.agent_ids:
            actor_params += list(self.actors[a].parameters())
        self.optim_actor = torch.optim.Adam(actor_params, lr=self.learning_rate)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

    def _select_actions(self, obs_dict):
        action_dict = {}
        logp_dict = {}
        value_inputs = []
        masks = {}
        if self.use_action_masks and hasattr(self.env, "get_action_masks"):
            try:
                masks = self.env.get_action_masks() or {}
            except Exception:
                masks = {}

        # build centralized obs vector
        central_obs = np.concatenate([obs_dict[a].astype(np.float32).ravel() for a in self.agent_ids]).astype(np.float32)
        central_obs_t = torch.tensor(central_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        # critic value
        with torch.no_grad():
            value = self.critic(central_obs_t).cpu().numpy().item()

        for a in self.agent_ids:
            obs = obs_dict[a]
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits = self.actors[a](obs_t)
            m = masks.get(a, None)
            if m is not None:
                mask_t = torch.tensor(np.asarray(m, dtype=np.float32), dtype=torch.float32, device=DEVICE).unsqueeze(0)
                logits = logits + (mask_t - 1.0) * 1e9
            dist = Categorical(logits=logits)
            act_t = dist.sample()
            action = int(act_t.detach().cpu().numpy().item())

            if self.exploration_eps > 0.0 and np.random.rand() < self.exploration_eps:
                if m is not None:
                    valid = np.flatnonzero(np.asarray(m, dtype=np.float32) > 0.5)
                else:
                    valid = np.arange(int(logits.shape[-1]), dtype=np.int64)
                if valid.size > 0:
                    action = int(np.random.choice(valid))
                    act_t = torch.tensor([action], dtype=torch.long, device=DEVICE)

            logp = dist.log_prob(act_t).detach().cpu().numpy().item()
            action_dict[a] = int(action)
            logp_dict[a] = float(logp)
        return action_dict, logp_dict, value, central_obs

    def train(self):
        obs_dict = self.env.reset()
        ep_rewards = deque(maxlen=100)
        episode_reward = defaultdict(float)

        while self.global_step < self.total_timesteps:
            # rollout collection
            for step in range(self.n_steps):
                action_dict, logp_dict, value, central_obs = self._select_actions(obs_dict)

                next_obs, reward_dict, done_dict, info_dict = self.env.step(action_dict)
                # note: env returns dones with '__all__'

                # store transitions per agent
                for a in self.agent_ids:
                    self.buffers[a].append(
                        obs=obs_dict[a].copy(),
                        action=action_dict[a],
                        logp=logp_dict[a],
                        reward=reward_dict.get(a, 0.0),
                        done=done_dict.get(a, False),
                        value=value,  # centralized value for this timestep
                        central_obs=central_obs.copy(),
                    )
                    episode_reward[a] += reward_dict.get(a, 0.0)

                obs_dict = next_obs
                self.global_step += 1

                # handle episode termination logging
                if done_dict.get("__all__", False):
                    # sum episode reward across agents
                    total_ep_reward = sum([episode_reward[a] for a in self.agent_ids])
                    ep_rewards.append(total_ep_reward)
                    self.writer.add_scalar("episode/total_reward", total_ep_reward, self.episode_count)
                    print(f"[Episode {self.episode_count}] total_reward={total_ep_reward:.2f} global_step={self.global_step}")
                    # reset counters
                    episode_reward = defaultdict(float)
                    self.episode_count += 1
                    obs_dict = self.env.reset()

                # save checkpoints occasionally
                if self.global_step % self.save_freq == 0:
                    self._save_checkpoint(f"checkpoint_{self.global_step}.pt")

                if self.global_step >= self.total_timesteps:
                    break

            # After collecting rollout of n_steps, compute returns & advantages per agent
            # We will compute advantage per agent using stored rewards and the centralized values.
            # For bootstrap last value, compute critic on last central_obs
            last_central_obs = np.concatenate([obs_dict[a].astype(np.float32).ravel() for a in self.agent_ids]).astype(np.float32)
            last_value = self.critic(torch.tensor(last_central_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)).cpu().detach().numpy().item()

            # prepare training batches aggregated across agents
            dataset = []
            for a in self.agent_ids:
                buf = self.buffers[a]
                if len(buf) == 0:
                    continue
                # compute advantage & returns
                advs, returns = compute_gae(buf.rewards, buf.values, buf.dones, last_value, self.gamma, self.gae_lambda)
                # store tuples for minibatch sampling
                for i in range(len(buf)):
                    dataset.append(
                        {
                            "agent": a,
                            "obs": buf.obs[i],
                            "action": buf.actions[i],
                            "logp": buf.log_probs[i],
                            "return": returns[i],
                            "adv": advs[i],
                            "central_obs": buf.central_obs[i],
                        }
                    )
                # clear buffer
                buf.clear()

            if len(dataset) == 0:
                continue

            # normalize advantages
            advs_all = np.array([d["adv"] for d in dataset], dtype=np.float32)
            adv_mean, adv_std = advs_all.mean(), advs_all.std() + 1e-8
            for d in dataset:
                d["adv"] = (d["adv"] - adv_mean) / adv_std

            # convert dataset to indices and do PPO updates
            dataset_size = len(dataset)
            indices = np.arange(dataset_size)
            minibatch_size = max(1, min(self.batch_size, dataset_size // 4))

            for epoch in range(self.update_epochs):
                np.random.shuffle(indices)
                for start in range(0, dataset_size, minibatch_size):
                    batch_idx = indices[start : start + minibatch_size]
                    # collect tensors for this minibatch
                    actor_loss = 0.0
                    critic_loss = 0.0
                    entropy_loss = 0.0

                    # accumulate losses across samples in minibatch
                    for idx in batch_idx:
                        sample = dataset[idx]
                        a_id = sample["agent"]
                        obs = torch.tensor(sample["obs"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        act = torch.tensor(sample["action"], dtype=torch.long, device=DEVICE).unsqueeze(0)
                        old_logp = torch.tensor(sample["logp"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        ret = torch.tensor(sample["return"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        adv = torch.tensor(sample["adv"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        central_obs = torch.tensor(sample["central_obs"], dtype=torch.float32, device=DEVICE).unsqueeze(0)

                        # actor forward
                        logits = self.actors[a_id](obs)
                        dist = Categorical(logits=logits)
                        new_logp = dist.log_prob(act.squeeze(-1)).unsqueeze(0)
                        entropy = dist.entropy().mean()

                        ratio = torch.exp(new_logp - old_logp)
                        surr1 = ratio * adv
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
                        actor_loss_sample = -torch.min(surr1, surr2).mean()

                        # critic forward (centralized)
                        value_pred = self.critic(central_obs).unsqueeze(0)
                        critic_loss_sample = (ret - value_pred).pow(2).mean()

                        actor_loss += actor_loss_sample
                        critic_loss += critic_loss_sample
                        entropy_loss += entropy

                    # normalize by minibatch length
                    mb_len = float(len(batch_idx))
                    actor_loss = actor_loss / mb_len
                    critic_loss = critic_loss / mb_len
                    entropy_loss = entropy_loss / mb_len

                    # total loss
                    total_actor_loss = actor_loss - self.ent_coef * entropy_loss
                    total_critic_loss = self.vf_coef * critic_loss

                    # step actor optimizer
                    self.optim_actor.zero_grad()
                    total_actor_loss.backward()
                    nn.utils.clip_grad_norm_( [p for a in self.agent_ids for p in self.actors[a].parameters()], self.max_grad_norm)
                    self.optim_actor.step()

                    # step critic optimizer
                    self.optim_critic.zero_grad()
                    total_critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.optim_critic.step()

            # logging
            self.writer.add_scalar("train/steps", self.global_step, self.global_step)
            # sample actor/critic loss approximation logging could be added here
            if len(ep_rewards) > 0:
                self.writer.add_scalar("episode/avg_last_100_reward", float(np.mean(ep_rewards)), self.global_step)

        # finalize
        self._save_checkpoint("final.pt")
        self.writer.close()

    def _save_checkpoint(self, name="checkpoint.pt"):
        os.makedirs("logs/checkpoints", exist_ok=True)
        path = os.path.join("logs/checkpoints", name)
        payload = {
            "actors": {a: self.actors[a].state_dict() for a in self.agent_ids},
            "critic": self.critic.state_dict(),
            "optim_actor": self.optim_actor.state_dict(),
            "optim_critic": self.optim_critic.state_dict(),
            "global_step": self.global_step,
        }
        torch.save(payload, path)
        print(f"Saved checkpoint to {path}")


# ---------------------------
# Main
# ---------------------------
def load_config(path="configs/default.yaml"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    else:
        return {}


if __name__ == "__main__":
    cfg = load_config("configs/default.yaml")
    env = FactoryEnv(config_path="configs/default.yaml")
    trainer = MAPPOTrainer(env, cfg)
    trainer.train()
