# trainer/eval_and_render.py
import numpy as np
import time
from stable_baselines3 import PPO

from env.factory_env import FactoryEnv
from trainer.centralized_env import CentralizedEnv

def rollout(model_path, episodes=5, render=True):
    base = FactoryEnv(init_materials=12)
    env = CentralizedEnv(base)
    model = PPO.load(model_path)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        if render:
            import matplotlib.pyplot as plt
            plt.ion()
            fig = plt.figure(figsize=(8,8))

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            step += 1
            if render:
                env.render()
                time.sleep(0.05)
        print(f"Episode {ep}: steps={step}, reward={total_reward}, sink={env.multi_env.Z_count if hasattr(env.multi_env,'Z_count') else getattr(env.multi_env,'world').sink_count}")

    if render:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    rollout("models/ppo_centralized.zip", episodes=3, render=True)