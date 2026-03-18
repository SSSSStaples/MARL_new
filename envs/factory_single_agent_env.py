import gymnasium as gym

from .factory_env import FactoryEnv


class FactorySingleAgentEnv(gym.Env):
    """
    Gymnasium-compatible single-agent adapter around the multi-agent FactoryEnv.

    - Controls exactly one AGV (default: "agv_1")
    - Other AGVs follow a fixed policy (default: idle)
    - Exposes standard Gymnasium signatures:
        reset() -> (obs, info)
        step(a) -> (obs, reward, terminated, truncated, info)
    """

    metadata = {"render_modes": []}

    def __init__(self, config_path="configs/default.yaml", controlled_agent="agv_1", other_policy="idle"):
        super().__init__()
        self.multi_env = FactoryEnv(config_path=config_path)
        self.controlled_agent = str(controlled_agent)
        self.other_policy = str(other_policy).strip().lower()

        if self.controlled_agent not in self.multi_env.agent_ids:
            raise ValueError(
                f"controlled_agent={self.controlled_agent!r} not in agent_ids={self.multi_env.agent_ids}"
            )

        self.action_space = self.multi_env.get_action_space(self.controlled_agent)
        self.observation_space = self.multi_env.get_observation_space(self.controlled_agent)

    def reset(self, seed=None, options=None):
        obs_dict = self.multi_env.reset(seed=seed, options=options)
        obs = obs_dict[self.controlled_agent]
        info = {}
        return obs, info

    def _other_action(self, agent_id):
        if self.other_policy == "random":
            return int(self.multi_env.get_action_space(agent_id).sample())
        return 0  # idle

    def step(self, action):
        action_dict = {a: self._other_action(a) for a in self.multi_env.agent_ids}
        action_dict[self.controlled_agent] = int(action)

        obs_dict, reward_dict, done_dict, info_dict = self.multi_env.step(action_dict)

        obs = obs_dict[self.controlled_agent]
        reward = float(reward_dict.get(self.controlled_agent, 0.0))

        terminated = bool(self.multi_env.completed_orders >= self.multi_env.total_orders)
        truncated = bool(self.multi_env.step_count >= self.multi_env.max_steps and not terminated)

        info = dict(info_dict or {})
        info["multi_reward_dict"] = reward_dict
        info["multi_done_dict"] = done_dict
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.multi_env.render()

    def close(self):
        return self.multi_env.close()

