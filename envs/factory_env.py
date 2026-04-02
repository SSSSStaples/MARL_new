import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml

from .factory_env_action_mask import get_action_mask as _get_action_mask
from .factory_env_action_mask import get_action_masks as _get_action_masks
from .factory_env_agv import agv_target_position as _agv_target_position
from .factory_env_agv import apply_agv_actions as _apply_agv_actions
from .factory_env_agv import current_station as _current_station
from .factory_env_init import init_internal_state as _init_internal_state
from .factory_env_init import load_job_types as _load_job_types
from .factory_env_init import load_orders as _load_orders
from .factory_env_machine import machine_step as _machine_step
from .factory_env_obs import get_obs_dict as _get_obs_dict
from .factory_env_render import close as _close_render
from .factory_env_render import render as _render
from .reward import RewardCalculator
from .station_objects import Station


class FactoryEnv(gym.Env):
    """
    Multi-agent Factory Environment (AGVs + Machines).

    - AGVs transport orders between stations.
    - Each process machine is an agent that chooses which eligible queued order to process next.
    """

    metadata = {"render_modes": []}

    def __init__(self, config_path="configs/default.yaml"):
        super().__init__()
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        env_cfg = cfg.get("env", {}) or {}
        reward_cfg = cfg.get("reward", {}) or {}
        job_cfg = cfg.get("jobs", {}) or {}
        orders_cfg = cfg.get("orders", []) or []

        # Processes / stations (keep current defaults; can be moved into config later).
        self.processes = ["Turning", "Grinding", "Milling", "Drilling"]
        self.stations = ["SRC", "TURNING", "GRINDING", "MILLING", "DRILLING", "SINK"]
        self.process_to_station = {
            "Turning": "TURNING",
            "Grinding": "GRINDING",
            "Milling": "MILLING",
            "Drilling": "DRILLING",
        }
        self.station_to_process = {station: process for process, station in self.process_to_station.items()}

        # Core env params
        self.max_steps = int(env_cfg.get("max_steps", 300))
        self.agv_count = int(env_cfg.get("agv_count", 2))
        self.agv_speed = float(env_cfg.get("agv_speed", 0.2))
        self.pickup_radius = float(env_cfg.get("pickup_radius", 0.5))
        self.drop_radius = float(env_cfg.get("drop_radius", 0.5))
        self.boundary_margin = float(env_cfg.get("boundary_margin", 0.0))
        self.collision_radius = float(env_cfg.get("collision_radius", 0.0))
        self.arrival_shift_to_first = bool(env_cfg.get("arrival_shift_to_first", False))
        # Simplify the task by auto-delivering finished orders when an AGV reaches SINK.
        # This reduces the need to learn an extra "drop" decision at the final station.
        self.auto_deliver_at_sink = bool(env_cfg.get("auto_deliver_at_sink", True))
        # Training aid: when an AGV is carrying a finished order and is within SINK drop radius,
        # force the only valid action to be DROP. This avoids "arrive at SINK but never drop".
        self.force_drop_at_sink = bool(env_cfg.get("force_drop_at_sink", False))
        # Training aid: when an empty AGV is at a station with an unclaimed finished handoff request,
        # force PICK on that finished order (prevents "finished exists but never picked").
        self.force_pick_finished_handoff = bool(env_cfg.get("force_pick_finished_handoff", False))
        # Training aid: when an empty AGV is at any non-SINK station with pickable orders, force PICK (rank 0).
        self.force_pick_at_station = bool(env_cfg.get("force_pick_at_station", False))
        # Training aid: when an AGV is carrying an order and is within a valid drop station radius, force DROP.
        self.force_drop_when_valid = bool(env_cfg.get("force_drop_when_valid", False))

        # Machine input buffer capacity constraint:
        # count only eligible waiting orders: (not finished) and next_process == station_process
        self.eligible_buffer_capacity = int(env_cfg.get("eligible_buffer_capacity", 5))

        # Picking / scheduling options
        self.pickup_index_count = int(env_cfg.get("pickup_index_count", 1))
        self.pick_sort_key = str(env_cfg.get("pick_sort_key", "fifo")).strip().lower()

        # Observation feature toggles
        obs_cfg = env_cfg.get("obs", {}) or {}
        self.obs_station_radius = float(obs_cfg.get("station_radius", max(self.pickup_radius, self.drop_radius)))
        self.obs_include_current_station = bool(obs_cfg.get("include_current_station", False))
        self.obs_pick_window_k = int(obs_cfg.get("pick_window_k", 0))
        self.obs_include_machine_timer = bool(obs_cfg.get("include_machine_timer", False))
        self.obs_include_station_positions = bool(obs_cfg.get("include_station_positions", False))
        self.obs_station_pos_mode = str(obs_cfg.get("station_pos_mode", "relative")).strip().lower()
        self.obs_station_pos_normalize = bool(obs_cfg.get("station_pos_normalize", True))
        self.obs_include_neighbor_agvs = bool(obs_cfg.get("include_neighbor_agvs", False))
        self.obs_neighbor_k = int(obs_cfg.get("neighbor_k", 1))
        self.obs_neighbor_normalize = bool(obs_cfg.get("neighbor_normalize", True))
        self.obs_include_target_vector = bool(obs_cfg.get("include_target_vector", False))
        self.obs_target_normalize = bool(obs_cfg.get("target_normalize", True))
        # Handoff alert signals for AGVs: where finished orders are waiting, and how long they've waited.
        self.obs_include_handoff_alert = bool(obs_cfg.get("include_handoff_alert", False))
        self.obs_handoff_include_age = bool(obs_cfg.get("handoff_include_age", True))

        # Spatial bounds / station positions
        bounds = env_cfg.get("bounds", {"xmin": 0.0, "ymin": 0.0, "xmax": 15.0, "ymax": 15.0}) or {}
        self.bounds = {
            "xmin": float(bounds.get("xmin", 0.0)),
            "ymin": float(bounds.get("ymin", 0.0)),
            "xmax": float(bounds.get("xmax", 15.0)),
            "ymax": float(bounds.get("ymax", 15.0)),
        }
        self.station_positions = dict(env_cfg.get("station_positions", {}) or {})
        if not self.station_positions:
            self.station_positions = {
                "SRC": (1.0, 1.0),
                "TURNING": (4.0, 4.0),
                "GRINDING": (7.0, 4.0),
                "MILLING": (10.0, 4.0),
                "DRILLING": (13.0, 4.0),
                "SINK": (13.0, 1.0),
            }
        # Ensure all stations exist (fallback to (0,0) if missing).
        for s in self.stations:
            self.station_positions.setdefault(s, (0.0, 0.0))

        # Process time defaults (used by machine + observation normalization)
        self.process_time_defaults = dict(env_cfg.get("process_time_defaults", {}) or {})
        if not self.process_time_defaults:
            self.process_time_defaults = {"Turning": 30, "Grinding": 20, "Milling": 40, "Drilling": 10}

        # Reward calculator + shaping mode
        self.rewarder = RewardCalculator(reward_cfg)
        self.global_task_complete_reward = bool(reward_cfg.get("global_task_complete", False))
        self.distance_shaping_mode = str(reward_cfg.get("distance_shaping_mode", "progress")).strip().lower()

        # Machines are agents (your setting)
        self.machine_as_agent = bool(env_cfg.get("machine_as_agent", True))
        self.machine_action_k = int(env_cfg.get("machine_action_k", 3))
        self.machine_obs_k = int(env_cfg.get("machine_obs_k", self.machine_action_k))
        self.machine_sort_key = str(env_cfg.get("machine_sort_key", "fifo")).strip().lower()

        # Agents list
        self.agv_ids = [f"agv_{i+1}" for i in range(self.agv_count)]
        self.machine_ids = [f"machine_{p}" for p in self.processes] if self.machine_as_agent else []
        self.process_to_machine_id = {p: f"machine_{p}" for p in self.processes}
        self.agent_ids = list(self.agv_ids) + list(self.machine_ids)

        # Action spaces
        self._pick_action_start = 9
        self._pick_action_end = self._pick_action_start + max(1, self.pickup_index_count) - 1
        self._drop_action = self._pick_action_end + 1

        self.action_spaces = {a: spaces.Discrete(self._drop_action + 1) for a in self.agv_ids}
        if self.machine_as_agent:
            for mid in self.machine_ids:
                self.action_spaces[mid] = spaces.Discrete(int(self.machine_action_k) + 1)

        # Observation spaces (dim formula matches envs/factory_env_obs.py)
        agv_obs_dim = (
            2  # pos
            + 2  # vel
            + 1  # carry_flag
            + len(self.processes)  # carry_next_process
            + len(_load_job_types(job_cfg))  # carry_job
            + len(self.stations)  # queue_lens
            + len(self.processes)  # machine_busy
            + 1  # time_frac
        )
        if self.obs_include_station_positions:
            agv_obs_dim += 2 * len(self.stations)
        if self.obs_include_neighbor_agvs:
            agv_obs_dim += 2 * int(max(0, self.obs_neighbor_k))
        if self.obs_include_target_vector:
            agv_obs_dim += 2
        if self.obs_include_machine_timer:
            agv_obs_dim += len(self.processes)
        if self.obs_include_current_station:
            agv_obs_dim += len(self.stations)
        if self.obs_include_handoff_alert:
            # Per non-SINK station: finished_exists flag (0/1) and optional waiting age feature.
            non_sink_n = int(max(0, len(self.stations) - 1))
            agv_obs_dim += non_sink_n
            if self.obs_handoff_include_age:
                agv_obs_dim += non_sink_n
        if self.obs_pick_window_k > 0:
            per_order_dim = 1 + 1 + len(self.processes)  # due_remaining, priority, next_process onehot
            agv_obs_dim += int(self.obs_pick_window_k) * int(per_order_dim)

        machine_obs_dim = 0
        if self.machine_as_agent:
            per_order = 1 + 1 + 1  # due_remaining, priority, proc_time_norm
            machine_obs_dim = 1 + 1 + 1 + 1 + int(self.machine_obs_k) * int(per_order)

        self.observation_spaces = {}
        for a in self.agv_ids:
            self.observation_spaces[a] = spaces.Box(low=-np.inf, high=np.inf, shape=(int(agv_obs_dim),), dtype=np.float32)
        for m in self.machine_ids:
            self.observation_spaces[m] = spaces.Box(low=-np.inf, high=np.inf, shape=(int(machine_obs_dim),), dtype=np.float32)

        # Job types / orders template
        self.job_types = _load_job_types(job_cfg)
        self.orders_template = _load_orders(orders_cfg)
        self.total_orders = len(self.orders_template)

        # For obs normalization of process times
        default_max_t = max([int(v) for v in self.process_time_defaults.values()] or [1])
        orders_max_t = 0
        for o in self.orders_template:
            if isinstance(o, dict):
                pts = o.get("process_times", []) or []
                if pts:
                    try:
                        orders_max_t = max(orders_max_t, max(int(x) for x in pts))
                    except Exception:
                        pass
        cfg_max_t = int(env_cfg.get("process_time_max", 0) or 0)
        self.process_time_max = int(max(1, default_max_t, orders_max_t, cfg_max_t))

        # World objects
        self.station_objs = {s: Station(s, self.station_positions[s]) for s in self.stations}
        self.machines = {p: {"busy": False, "timer": 0, "timer_init": 0, "order": None} for p in self.processes}

        # Internal state
        _init_internal_state(self)
        self._viz = {"fig": None, "ax": None, "last_step": -1}

    # -------------------
    # Gym-style helpers
    # -------------------
    def get_action_space(self, agent_id):
        return self.action_spaces[agent_id]

    def get_observation_space(self, agent_id):
        return self.observation_spaces[agent_id]

    def get_action_mask(self, agent_id):
        return _get_action_mask(self, agent_id)

    def get_action_masks(self):
        return _get_action_masks(self)

    # -------------------
    # Compatibility helpers used by split modules
    # -------------------
    def _agv_target_position(self, agv_index, carry_before):
        return _agv_target_position(self, agv_index, carry_before)

    def _current_station(self, pos, radius):
        return _current_station(self, pos, radius)

    def _get_obs_dict(self):
        return _get_obs_dict(self)

    # -------------------
    # Core API
    # -------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        _init_internal_state(self)
        return self._get_obs_dict()

    def step(self, action_dict):
        self.step_count += 1
        events = {
            "agv_out_of_bounds": [],
            "agv_wrong_drop": [],
            "agv_buffer_full": [],
            "agv_pick_failed": [],
            "agv_drop_failed": [],
            "agv_delivered": [],
            "agv_picked": [],
            "agv_picked_finished": [],
            "agv_correct_drop": [],
            "machine_started": [],
            "machine_completed": [],
            "machine_order_advanced": [],
            "machine_order_finished": [],
            "handoff_requested": [],
            "handoff_claimed": [],
            "machine_select_failed": [],
        }

        # spawn arriving orders
        while self.pending_orders and self.pending_orders[0].arrival_time <= self.step_count:
            order = self.pending_orders.pop(0)
            self.station_objs["SRC"].add_item(order)

        # AGV actions + movement shaping + collisions
        _apply_agv_actions(self, action_dict or {}, events)
        for d in events.get("agv_delivered", []) or []:
            oid = d.get("order_id", None)
            if oid is not None:
                self.delivered_order_ids.add(int(oid))

        # Machine processing (machines are agents)
        selections, machine_events = _machine_step(self, action_dict or {})
        for k in (
            "machine_started",
            "machine_completed",
            "machine_order_advanced",
            "machine_order_finished",
            "handoff_requested",
            "machine_select_failed",
        ):
            if k in machine_events:
                events.setdefault(k, []).extend(machine_events.get(k, []))

        # Order status matrix (orders x [Turning,Grinding,Milling,Drilling,SINK]).
        # Values: -1 not in route, 0 not done, 1 done.
        order_objs = {}
        for s in self.stations:
            st = self.station_objs[s]
            for item in st.queue:
                if hasattr(item, "order_id"):
                    order_objs[int(item.order_id)] = item
        for item in self.agv_carry:
            if item is not None and hasattr(item, "order_id"):
                order_objs[int(item.order_id)] = item
        for p in self.processes:
            mo = self.machines[p].get("order", None)
            if mo is not None and hasattr(mo, "order_id"):
                order_objs[int(mo.order_id)] = mo

        order_ids = sorted(
            {
                int(o.get("id"))
                for o in (self.orders_template or [])
                if isinstance(o, dict) and (o.get("id", None) is not None)
            }
        )
        cols = list(self.processes) + ["SINK"]
        matrix = []
        for oid in order_ids:
            delivered = 1 if int(oid) in getattr(self, "delivered_order_ids", set()) else 0
            o = order_objs.get(int(oid), None)
            if o is not None:
                route = list(getattr(o, "route", []) or [])
                step_idx = int(getattr(o, "step_idx", 0) or 0)
            else:
                # fallback to template route
                route = []
                for od in (self.orders_template or []):
                    if isinstance(od, dict) and int(od.get("id", -1)) == int(oid):
                        jt = od.get("job_type", None)
                        if jt in self.job_types:
                            route = list(self.job_types[jt].get("route", []) or [])
                        break
                step_idx = len(route) if delivered else 0

            row = []
            for proc in self.processes:
                if proc not in route:
                    row.append(-1)
                    continue
                idx = route.index(proc)
                row.append(1 if step_idx > idx else 0)
            row.append(int(delivered))
            matrix.append(row)

        # Finished-order waiting signals:
        # After the last machine finishes an order, it is placed into that station queue and must be
        # picked up by an AGV and delivered to SINK. Penalize letting finished orders sit around.
        finished_waiting = 0
        finished_unclaimed_waiting = 0
        for s in self.stations:
            if s == "SINK":
                continue
            station_obj = self.station_objs[s]
            for item in station_obj.queue:
                if getattr(item, "finished", False):
                    finished_waiting += 1
                    if getattr(item, "reserved_by", None) is None:
                        finished_unclaimed_waiting += 1
        finished_carried = int(sum(1 for x in self.agv_carry if getattr(x, "finished", False)))
        events["finished_waiting_orders"] = int(finished_waiting)
        events["finished_unclaimed_waiting_orders"] = int(finished_unclaimed_waiting)
        events["finished_carried_orders"] = int(finished_carried)

        # reward
        rewards = self.rewarder.compute(self.agent_ids, events)

        # done condition
        done = bool(self.step_count >= self.max_steps or self.completed_orders >= self.total_orders)
        dones = {a: done for a in self.agent_ids}
        dones["__all__"] = done

        obs = self._get_obs_dict()
        infos = {
            "completed_orders": int(self.completed_orders),
            "time": int(self.step_count),
            "machine_selections": selections,
            "collision_agents": list(events.get("collision_agents", [])),
            "delivered": list(events.get("agv_delivered", [])),
            "out_of_bounds_agents": list(events.get("agv_out_of_bounds", [])),
            "src_queue_len": int(self.station_objs["SRC"].queue_length()),
            "agv_carry_count": int(sum(1 for x in self.agv_carry if x is not None)),
            "picked_agents": list(events.get("agv_picked", [])),
            "picked_finished_agents": list(events.get("agv_picked_finished", [])),
            "correct_drop_agents": list(events.get("agv_correct_drop", [])),
            "wrong_drop_agents": list(events.get("agv_wrong_drop", [])),
            "buffer_full_agents": list(events.get("agv_buffer_full", [])),
            "pick_failed_agents": list(events.get("agv_pick_failed", [])),
            "drop_failed_agents": list(events.get("agv_drop_failed", [])),
            "machine_started": list(events.get("machine_started", [])),
            "machine_completed": list(events.get("machine_completed", [])),
            "machine_order_advanced": list(events.get("machine_order_advanced", [])),
            "machine_order_finished": list(events.get("machine_order_finished", [])),
            "handoff_requested": list(events.get("handoff_requested", [])),
            "handoff_claimed": list(events.get("handoff_claimed", [])),
            "machine_select_failed": list(events.get("machine_select_failed", [])),
            "finished_waiting_orders": int(events.get("finished_waiting_orders", 0) or 0),
            "finished_unclaimed_waiting_orders": int(events.get("finished_unclaimed_waiting_orders", 0) or 0),
            "finished_carried_orders": int(events.get("finished_carried_orders", 0) or 0),
            "order_status_order_ids": order_ids,
            "order_status_cols": cols,
            "order_status_matrix": matrix,
        }
        return obs, rewards, dones, infos

    # -------------------
    # Render
    # -------------------
    def render(self):
        return _render(self)

    def close(self):
        return _close_render(self)
