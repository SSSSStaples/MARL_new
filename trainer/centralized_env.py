# trainer/centralized_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CentralizedEnv(gym.Env):
    """
    Wrap multi-agent FactoryEnv into a single-agent environment for PPO.

    Input:
        FactoryEnv (returns dict obs and dict rewards)

    Output:
        observation: concatenated vector (mover1 + mover2)
        action: MultiDiscrete([7, 7])
        reward: scalar (sum of both agents)
    """

    def __init__(self, multi_env):

        super().__init__()

        self.multi_env = multi_env

        # -------- Action Space --------
        # mover1: 7 actions
        # mover2: 7 actions
        self.action_space = spaces.MultiDiscrete([7, 7])

        # -------- Observation Space --------
        # each mover obs = 19 dims (我们之前定义的)
        obs_dim_single = multi_env.observation_space["mover1"].shape[0]
        total_dim = obs_dim_single * 2

        self.observation_space = spaces.Box(
            low=-1e6,
            high=1e6,
            shape=(total_dim,),
            dtype=np.float32
        )

    # -----------------------------------------------------

    def reset(self, seed=None, options=None):

        obs_dict, _ = self.multi_env.reset(seed=seed)

        flat_obs = self._flatten_obs(obs_dict)

        return flat_obs, {}

    # -----------------------------------------------------

    def step(self, action):

        # action = [a1, a2]
        a1 = int(action[0])
        a2 = int(action[1])

        obs_dict, rewards_dict, terminated, truncated, info = self.multi_env.step({
            "mover1": a1,
            "mover2": a2
        })

        # sum rewards
        reward = float(sum(rewards_dict.values()))

        flat_obs = self._flatten_obs(obs_dict)

        done = terminated or truncated

        return flat_obs, reward, done, False, info

    # -----------------------------------------------------

    def _flatten_obs(self, obs_dict):

        obs1 = np.array(obs_dict["mover1"], dtype=np.float32)
        obs2 = np.array(obs_dict["mover2"], dtype=np.float32)

        return np.concatenate([obs1, obs2])

    # -----------------------------------------------------

    def render(self):
        return self.multi_env.render()

    def close(self):
        if hasattr(self.multi_env, "close"):
            self.multi_env.close()