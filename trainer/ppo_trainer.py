# trainer/ppo_trainer.py

import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from env.factory_env import FactoryEnv
from trainer.centralized_env import CentralizedEnv


class PPOTrainer:

    def __init__(
        self,
        total_timesteps=300_000,
        learning_rate=3e-4,
        save_dir="models"
    ):

        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        # ---- Create Environment ----
        self.env = DummyVecEnv([self._make_env])

        # ---- Build PPO model ----
        policy_kwargs = dict(
            net_arch=dict(
                pi=[128, 128],
                vf=[128, 128]
            )
        )

        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.learning_rate,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1,
            policy_kwargs=policy_kwargs
        )

    def _make_env(self):
        base_env = FactoryEnv()
        return CentralizedEnv(base_env)

    # ---------------- TRAIN ----------------

    def train(self):

        checkpoint_callback = CheckpointCallback(
            save_freq=50_000,
            save_path=self.save_dir,
            name_prefix="ppo_factory"
        )

        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=checkpoint_callback
        )

        self.model.save(os.path.join(self.save_dir, "ppo_final"))
        print("Training finished.")

    # ---------------- EVALUATE ----------------

    def evaluate(self, episodes=5):

        base_env = FactoryEnv()
        env = CentralizedEnv(base_env)

        for ep in range(episodes):

            obs, _ = env.reset()
            done = False
            total_reward = 0
            step = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward
                step += 1

            print(f"Episode {ep}:")
            print("  Steps:", step)
            print("  Total reward:", total_reward)
            print("  Sink count:", base_env.world.sink_count)