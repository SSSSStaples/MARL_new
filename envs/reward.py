class RewardCalculator:
    def __init__(self, reward_cfg):
        self.r_task = float(reward_cfg.get("task_complete", 50.0))
        self.r_time = float(reward_cfg.get("time_penalty", -0.1))
        self.time_penalty_mode = str(reward_cfg.get("time_penalty_mode", "per_agent")).strip().lower()
        self.r_oob = float(reward_cfg.get("out_of_bounds_penalty", -5.0))
        self.r_wrong_drop = float(reward_cfg.get("wrong_drop_penalty", -1.0))
        self.r_late_per_step = float(reward_cfg.get("late_penalty_per_step", -0.2))
        self.r_collision = float(reward_cfg.get("collision_penalty", -2.0))
        self.w_distance = float(reward_cfg.get("distance_reward_weight", 0.0))
        self.w_precision = float(reward_cfg.get("precision_reward_weight", 0.0))
        self.w_boundary = float(reward_cfg.get("boundary_shaping_weight", 0.0))

    def compute(self, agent_ids, events):
        rewards = {a: 0.0 for a in agent_ids}

        for a in events.get("collision_agents", []):
            rewards[a] += self.r_collision

        for a in events.get("agv_out_of_bounds", []):
            rewards[a] += self.r_oob

        for a in events.get("agv_wrong_drop", []):
            rewards[a] += self.r_wrong_drop

        global_completions = int(events.get("global_task_complete", 0) or 0)
        if global_completions > 0:
            for a in agent_ids:
                rewards[a] += self.r_task * float(global_completions)

        for completion in events.get("agv_delivered", []):
            agent_id = completion["agent"]
            late_steps = completion.get("late_steps", 0)
            # If global completion is enabled, only apply lateness to the delivering agent.
            if global_completions > 0:
                rewards[agent_id] += self.r_late_per_step * max(0, late_steps)
            else:
                rewards[agent_id] += self.r_task + self.r_late_per_step * max(0, late_steps)

        # Optional shaping signals
        dist_shaping = events.get("distance_shaping", {})
        if self.w_distance != 0.0 and isinstance(dist_shaping, dict):
            for agent_id, delta in dist_shaping.items():
                if agent_id in rewards:
                    rewards[agent_id] += self.w_distance * float(delta)

        precision_bonus = events.get("precision_bonus", {})
        if self.w_precision != 0.0 and isinstance(precision_bonus, dict):
            for agent_id, bonus in precision_bonus.items():
                if agent_id in rewards:
                    rewards[agent_id] += self.w_precision * float(bonus)

        boundary_shaping = events.get("boundary_shaping", {})
        if self.w_boundary != 0.0 and isinstance(boundary_shaping, dict):
            for agent_id, val in boundary_shaping.items():
                if agent_id in rewards:
                    rewards[agent_id] += self.w_boundary * float(val)

        for a in agent_ids:
            if self.time_penalty_mode in {"shared", "global"}:
                rewards[a] += float(self.r_time) / float(max(1, len(agent_ids)))
            else:
                rewards[a] += float(self.r_time)

        return rewards
