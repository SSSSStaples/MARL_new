class RewardCalculator:
    def __init__(self, reward_cfg):
        self.r_task = float(reward_cfg.get("task_complete", 50.0))
        self.r_time = float(reward_cfg.get("time_penalty", -0.1))
        self.r_collision = float(reward_cfg.get("collision_penalty", -10.0))
        self.r_oob = float(reward_cfg.get("out_of_bounds_penalty", -5.0))
        self.r_distance = float(reward_cfg.get("distance_shaping", 0.5))

    def compute(self, agent_ids, events):
        rewards = {a: 0.0 for a in agent_ids}

        if events.get("mover1_deliver_to_Y"):
            rewards["mover_1"] += self.r_task * 0.5

        if events.get("mover2_deliver_product_to_Z"):
            rewards["mover_2"] += self.r_task

        if events.get("mover2_drop_raw_to_Z"):
            rewards["mover_2"] += -1.0

        if events.get("manuf_started"):
            rewards["manuf_1"] += 1.0

        if events.get("manuf_finished"):
            rewards["manuf_1"] += 2.0

        if events.get("mover1_out_of_bounds"):
            rewards["mover_1"] += self.r_oob

        if events.get("mover2_out_of_bounds"):
            rewards["mover_2"] += self.r_oob

        if "mover1_to_Y_progress" in events:
            rewards["mover_1"] += max(0.0, events["mover1_to_Y_progress"]) * self.r_distance * 0.01

        if "mover2_to_Z_progress" in events:
            rewards["mover_2"] += max(0.0, events["mover2_to_Z_progress"]) * self.r_distance * 0.01

        for a in agent_ids:
            rewards[a] += self.r_time

        return rewards
