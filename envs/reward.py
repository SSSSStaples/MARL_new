class RewardCalculator:
    def __init__(self, reward_cfg):
        self.r_task = float(reward_cfg.get("task_complete", 50.0))
        self.r_time = float(reward_cfg.get("time_penalty", -0.1))
        self.time_penalty_mode = str(reward_cfg.get("time_penalty_mode", "per_agent")).strip().lower()
        self.r_oob = float(reward_cfg.get("out_of_bounds_penalty", -5.0))
        self.r_wrong_drop = float(reward_cfg.get("wrong_drop_penalty", -1.0))
        self.r_buffer_full = float(reward_cfg.get("buffer_full_penalty", 0.0))
        self.r_pick_failed = float(reward_cfg.get("pick_failed_penalty", 0.0))
        self.r_drop_failed = float(reward_cfg.get("drop_failed_penalty", 0.0))
        self.r_pick = float(reward_cfg.get("pick_reward", 0.0))
        self.r_pick_finished = float(reward_cfg.get("pick_finished_reward", 0.0))
        self.r_correct_drop = float(reward_cfg.get("correct_drop_reward", 0.0))
        # Early completion bonus at SINK delivery:
        # bonus = early_bonus_per_step * max(0, due_time - delivered_step)
        self.r_early_per_step = float(reward_cfg.get("early_bonus_per_step", 0.0))
        # Extra bonus if delivered on-time (late_steps == 0)
        self.r_on_time = float(reward_cfg.get("on_time_bonus", 0.0))
        self.r_late_per_step = float(reward_cfg.get("late_penalty_per_step", -0.2))
        self.r_collision = float(reward_cfg.get("collision_penalty", -2.0))
        self.r_machine_start = float(reward_cfg.get("machine_start_reward", 0.0))
        self.r_machine_complete = float(reward_cfg.get("machine_complete_reward", 0.0))
        self.r_machine_order_advance = float(reward_cfg.get("machine_order_advance_reward", 0.0))
        self.r_machine_order_finish = float(reward_cfg.get("machine_order_finish_reward", 0.0))
        self.r_machine_select_failed = float(reward_cfg.get("machine_select_failed_penalty", 0.0))
        self.r_handoff_claimed = float(reward_cfg.get("handoff_claimed_reward", 0.0))

        # Encourage picking up finished items and delivering them to SINK quickly.
        # Penalize each step that a finished order is waiting in a non-SINK station queue.
        # Mode:
        # - "agv": apply only to agents whose id starts with "agv_"
        # - "all": apply to all agents (AGVs + machines)
        self.r_finished_wait = float(reward_cfg.get("finished_wait_penalty_per_step", 0.0))
        self.r_finished_carried = float(reward_cfg.get("finished_carried_penalty_per_step", 0.0))
        self.finished_penalty_mode = str(reward_cfg.get("finished_penalty_mode", "agv")).strip().lower()

        # Optional global/team rewards (distributed to all agents)
        self.r_global_machine_start = float(reward_cfg.get("global_machine_start_reward", 0.0))
        self.r_global_machine_complete = float(reward_cfg.get("global_machine_complete_reward", 0.0))
        self.r_global_machine_order_advance = float(reward_cfg.get("global_machine_order_advance_reward", 0.0))
        self.r_global_machine_order_finish = float(reward_cfg.get("global_machine_order_finish_reward", 0.0))
        # Penalize the whole team for collisions (per step where any collision happens).
        self.r_global_collision = float(reward_cfg.get("global_collision_penalty", 0.0))
        self.w_distance = float(reward_cfg.get("distance_reward_weight", 0.0))
        self.w_precision = float(reward_cfg.get("precision_reward_weight", 0.0))
        self.w_boundary = float(reward_cfg.get("boundary_shaping_weight", 0.0))

    def compute(self, agent_ids, events):
        rewards = {a: 0.0 for a in agent_ids}

        def _targets():
            if self.finished_penalty_mode in {"all", "shared", "global"}:
                return agent_ids
            # default: AGV only
            return [a for a in agent_ids if str(a).startswith("agv_")]

        for a in events.get("collision_agents", []):
            rewards[a] += self.r_collision

        for a in events.get("agv_out_of_bounds", []):
            rewards[a] += self.r_oob

        for a in events.get("agv_wrong_drop", []):
            rewards[a] += self.r_wrong_drop

        for a in events.get("agv_buffer_full", []):
            rewards[a] += self.r_buffer_full

        for a in events.get("agv_pick_failed", []):
            rewards[a] += self.r_pick_failed

        for a in events.get("agv_drop_failed", []):
            rewards[a] += self.r_drop_failed

        for a in events.get("agv_picked", []):
            rewards[a] += self.r_pick

        for a in events.get("agv_picked_finished", []):
            rewards[a] += self.r_pick_finished

        for a in events.get("agv_correct_drop", []):
            rewards[a] += self.r_correct_drop

        for mid in events.get("machine_started", []):
            if mid in rewards:
                rewards[mid] += self.r_machine_start

        for mid in events.get("machine_completed", []):
            if mid in rewards:
                rewards[mid] += self.r_machine_complete

        for mid in events.get("machine_select_failed", []):
            if mid in rewards:
                rewards[mid] += self.r_machine_select_failed

        for item in events.get("machine_order_advanced", []):
            mid = item.get("machine_id") if isinstance(item, dict) else item
            if mid in rewards:
                rewards[mid] += self.r_machine_order_advance

        for item in events.get("machine_order_finished", []):
            mid = item.get("machine_id") if isinstance(item, dict) else item
            if mid in rewards:
                rewards[mid] += self.r_machine_order_finish

        # Finished orders waiting / being carried (delivery urgency shaping)
        waiting_n = int(events.get("finished_unclaimed_waiting_orders", events.get("finished_waiting_orders", 0)) or 0)
        if self.r_finished_wait != 0.0 and waiting_n > 0:
            for a in _targets():
                rewards[a] += self.r_finished_wait * float(waiting_n)

        carried_n = int(events.get("finished_carried_orders", 0) or 0)
        if self.r_finished_carried != 0.0 and carried_n > 0:
            for a in _targets():
                rewards[a] += self.r_finished_carried * float(carried_n)

        for item in events.get("handoff_claimed", []) or []:
            if not isinstance(item, dict):
                continue
            agv = item.get("agent", None)
            if agv in rewards:
                rewards[agv] += self.r_handoff_claimed

        # Global/team rewards (helps align AGV+machine coordination)
        if self.r_global_machine_start != 0.0:
            n = int(len(events.get("machine_started", []) or []))
            if n > 0:
                for a in agent_ids:
                    rewards[a] += self.r_global_machine_start * float(n)

        if self.r_global_machine_complete != 0.0:
            n = int(len(events.get("machine_completed", []) or []))
            if n > 0:
                for a in agent_ids:
                    rewards[a] += self.r_global_machine_complete * float(n)

        if self.r_global_machine_order_advance != 0.0:
            n = int(len(events.get("machine_order_advanced", []) or []))
            if n > 0:
                for a in agent_ids:
                    rewards[a] += self.r_global_machine_order_advance * float(n)

        if self.r_global_machine_order_finish != 0.0:
            n = int(len(events.get("machine_order_finished", []) or []))
            if n > 0:
                for a in agent_ids:
                    rewards[a] += self.r_global_machine_order_finish * float(n)

        if self.r_global_collision != 0.0 and (events.get("collision_agents") or events.get("collision_count", 0)):
            for a in agent_ids:
                rewards[a] += self.r_global_collision

        global_completions = int(events.get("global_task_complete", 0) or 0)
        if global_completions > 0:
            for a in agent_ids:
                rewards[a] += self.r_task * float(global_completions)

        for completion in events.get("agv_delivered", []):
            agent_id = completion["agent"]
            late_steps = completion.get("late_steps", 0)
            delivered_step = completion.get("delivered_step", None)
            due_time = completion.get("due_time", None)

            if (
                self.r_early_per_step != 0.0
                and delivered_step is not None
                and due_time is not None
            ):
                try:
                    early_steps = max(0, int(due_time) - int(delivered_step))
                except Exception:
                    early_steps = 0
                rewards[agent_id] += self.r_early_per_step * float(early_steps)

            # If global completion is enabled, only apply lateness to the delivering agent.
            if global_completions > 0:
                rewards[agent_id] += self.r_late_per_step * max(0, late_steps)
            else:
                rewards[agent_id] += self.r_task + self.r_late_per_step * max(0, late_steps)

            if self.r_on_time != 0.0 and int(late_steps) <= 0:
                rewards[agent_id] += self.r_on_time

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
