import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml

from .station_objects import Station, Order
from .reward import RewardCalculator


class FactoryEnv(gym.Env):
    """
    Multi-agent Factory Environment (multi-process, task-driven).

    Agents:
      - agv_1..agv_N : transport orders between stations

    Multi-agent API:
      obs, rewards, dones, infos = env.step(action_dict)
      obs: dict agent_id -> np.array 
      dones: dict agent_id -> bool, plus '__all__' key
      infos: dict with summary fields
    """

    metadata = {"render_modes": []}

    def __init__(self, config_path="configs/default.yaml"):
        super().__init__()
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        env_cfg = cfg.get("env", {})
        reward_cfg = cfg.get("reward", {})
        job_cfg = cfg.get("jobs", {})
        orders_cfg = cfg.get("orders", [])

        # core lists
        self.processes = ["Turning", "Grinding", "Milling", "Drilling"]
        self.stations = ["SRC", "TURNING", "GRINDING", "MILLING", "DRILLING", "SINK"]
        self.process_to_station = {
            "Turning": "TURNING",
            "Grinding": "GRINDING",
            "Milling": "MILLING",
            "Drilling": "DRILLING",
        }
        # Reverse mapping (only for process stations)
        self.station_to_process = {station: process for process, station in self.process_to_station.items()}

        # env params
        self.max_steps = int(env_cfg.get("max_steps", 300))
        self.agv_count = int(env_cfg.get("agv_count", 2))
        self.agv_speed = float(env_cfg.get("agv_speed", 0.2))
        self.pickup_radius = float(env_cfg.get("pickup_radius", 0.5))
        self.drop_radius = float(env_cfg.get("drop_radius", 0.5))

        # Optional: soft boundary shaping (discourage hugging walls).
        # - boundary_margin<=0 disables this shaping (default).
        self.boundary_margin = float(env_cfg.get("boundary_margin", 0.0))

        # Optional: collision modeling between AGVs (distance-based).
        # - collision_radius<=0 disables collision events (default).
        self.collision_radius = float(env_cfg.get("collision_radius", 0.0))
        # If enabled, shift all order arrival/due times so the first order arrives at t=1.
        # This avoids long "no-task" prefixes where agents wander away before any orders exist.
        self.arrival_shift_to_first = bool(env_cfg.get("arrival_shift_to_first", False))

        # Optional: allow agents to choose which queued order to pick.
        # - pickup_index_count=1 keeps the original (single "pick" action).
        # - pickup_index_count>1 expands the action space so that "pick" becomes K actions
        #   selecting the 0..K-1 ranked order within the current station queue.
        self.pickup_index_count = int(env_cfg.get("pickup_index_count", 1))
        self.pick_sort_key = str(env_cfg.get("pick_sort_key", "fifo")).strip().lower()

        # Optional: add scheduling-relevant features to observations.
        # - obs_include_current_station: append one-hot of current station (within obs_station_radius).
        # - obs_pick_window_k: append top-K order features at current station (ranked by pick_sort_key).
        # - obs_include_machine_timer: append per-process remaining timer fraction.
        # - obs_include_station_positions: append station position features (absolute or relative).
        # - obs_include_neighbor_agvs: append nearest-K other AGVs' relative positions.
        # - obs_include_target_vector: append 2D vector to the current shaping target (helps learn correct routing).
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
        bounds = env_cfg.get("bounds", {"xmin": 0.0, "ymin": 0.0, "xmax": 15.0, "ymax": 15.0})
        self.bounds = {
            "xmin": float(bounds.get("xmin", 0.0)),
            "ymin": float(bounds.get("ymin", 0.0)),
            "xmax": float(bounds.get("xmax", 15.0)),
            "ymax": float(bounds.get("ymax", 15.0)),
        }

        # process time defaults
        default_times = env_cfg.get("process_time_defaults", {})
        self.process_time_defaults = {
            "Turning": int(default_times.get("Turning", 30)),
            "Grinding": int(default_times.get("Grinding", 20)),
            "Milling": int(default_times.get("Milling", 40)),
            "Drilling": int(default_times.get("Drilling", 10)),
        }

        # stations
        default_positions = {
            "SRC": [1.0, 1.0],
            "TURNING": [4.0, 4.0],
            "GRINDING": [7.0, 4.0],
            "MILLING": [10.0, 4.0],
            "DRILLING": [13.0, 4.0],
            "SINK": [13.0, 1.0],
        }
        station_positions = env_cfg.get("station_positions", default_positions)
        self.station_positions = {k: station_positions.get(k, v) for k, v in default_positions.items()}
        self.station_objs = {name: Station(name, self.station_positions[name]) for name in self.stations}

        # machines per process (1 each)
        # timer_init tracks the initial duration of the currently-processing order (for stable normalization).
        self.machines = {p: {"busy": False, "timer": 0, "timer_init": 0, "order": None} for p in self.processes}

        # job types (from config or defaults)
        self.job_types = self._load_job_types(job_cfg)

        # orders (from config or defaults)
        self.orders_template = self._load_orders(orders_cfg)
        self.total_orders = len(self.orders_template)

        # Used to normalize processing-time features for machine agents.
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

        # reward calculator
        self.rewarder = RewardCalculator(reward_cfg)
        self.global_task_complete_reward = bool(reward_cfg.get("global_task_complete", False))
        # Distance shaping mode for movement-based dense rewards.
        # - "progress": only reward getting closer (no negative)
        # - "signed": reward (dist_prev - dist_new) (can be negative)
        self.distance_shaping_mode = str(reward_cfg.get("distance_shaping_mode", "progress")).strip().lower()

        # Optional: treat each process machine as an RL agent that chooses which eligible order to process next.
        self.machine_as_agent = bool(env_cfg.get("machine_as_agent", False))
        self.machine_action_k = int(env_cfg.get("machine_action_k", 3))
        self.machine_obs_k = int(env_cfg.get("machine_obs_k", self.machine_action_k))
        self.machine_sort_key = str(env_cfg.get("machine_sort_key", "fifo")).strip().lower()

        # agents list
        self.agv_ids = [f"agv_{i+1}" for i in range(self.agv_count)]
        self.machine_ids = [f"machine_{p}" for p in self.processes] if self.machine_as_agent else []
        self.process_to_machine_id = {p: f"machine_{p}" for p in self.processes}
        self.agent_ids = list(self.agv_ids) + list(self.machine_ids)

        # action spaces (per agent)
        # 0 idle
        # 1..8 move (N,S,W,E,NW,NE,SW,SE) with step size agv_speed
        # pick actions: 9..(9+pickup_index_count-1), drop: 9+pickup_index_count
        self._pick_action_start = 9
        self._pick_action_end = self._pick_action_start + max(1, self.pickup_index_count) - 1
        self._drop_action = self._pick_action_end + 1
        self.action_spaces = {a: spaces.Discrete(self._drop_action + 1) for a in self.agv_ids}
        if self.machine_as_agent:
            # action 0 => choose rank-0 by machine_sort_key (default)
            # action 1..K => choose rank-(action-1) (clipped to available candidates)
            for mid in self.machine_ids:
                self.action_spaces[mid] = spaces.Discrete(int(self.machine_action_k) + 1)

        # observation spaces
        # [pos(2), vel(2), carry_flag(1), carry_job(onehot), carry_next_process(onehot),
        #  station_pos(optional, 2*len(stations)), queue_lens(len(stations)), machine_busy(len(processes)),
        #  machine_timer(optional, len(processes)), time_frac(1)]
        agv_obs_dim = (
            2
            + 2
            + 1
            + len(self.job_types)
            + len(self.processes)
            + len(self.stations)
            + len(self.processes)
            + 1
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
        if self.obs_pick_window_k > 0:
            per_order_dim = 1 + 1 + len(self.processes)  # due_remaining, priority, next_process onehot
            agv_obs_dim += int(self.obs_pick_window_k) * int(per_order_dim)

        # Machine agent observation: [busy(1), timer_frac(1), eligible_q_len(1), time_frac(1), topK(due,priority,proc_time)]
        machine_obs_dim = 0
        if self.machine_as_agent:
            per_order = 1 + 1 + 1  # due_remaining, priority, proc_time_norm
            machine_obs_dim = 1 + 1 + 1 + 1 + int(self.machine_obs_k) * int(per_order)

        self.observation_spaces = {}
        for a in self.agv_ids:
            self.observation_spaces[a] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(int(agv_obs_dim),), dtype=np.float32
            )
        for m in self.machine_ids:
            self.observation_spaces[m] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(int(machine_obs_dim),), dtype=np.float32
            )
        
        # internal state
        self._init_internal_state()
        self._viz = {"fig": None, "ax": None, "last_step": -1}
     
    # -------------------
    # Init helpers
    # -------------------
    def _load_job_types(self, job_cfg):
        if job_cfg:
            return job_cfg
        return {
            "P1": {"part": "Shaft", "material": "Bar", "route": ["Turning", "Grinding"]},
            "P2": {"part": "Flange", "material": "Cube", "route": ["Turning", "Milling", "Drilling"]},
            "P3": {"part": "Plate", "material": "Plate", "route": ["Milling"]},
            "P4": {"part": "Frame", "material": "Bar", "route": ["Milling", "Drilling"]},
        }

    def _load_orders(self, orders_cfg):
        if orders_cfg:
            return orders_cfg
        return [
            {"id": 1, "job_type": "P1", "arrival": 15, "due": 160, "process_times": [35, 12]},
            {"id": 2, "job_type": "P2", "arrival": 50, "due": 180, "process_times": [32, 10, 20]},
            {"id": 3, "job_type": "P3", "arrival": 75, "due": 210, "process_times": [43]},
            {"id": 4, "job_type": "P4", "arrival": 100, "due": 230, "process_times": [38, 10]},
            {"id": 5, "job_type": "P1", "arrival": 130, "due": 260, "process_times": [47, 15]},
        ]

    def _init_internal_state(self):
        # agent positions and carry
        self.agv_positions = [np.array(self.station_positions["SRC"], dtype=np.float32) for _ in self.agv_ids]
        self.agv_velocities = [np.zeros(2, dtype=np.float32) for _ in self.agv_ids]
        self.agv_carry = [None for _ in self.agv_ids]

        # reset stations
        for s in self.station_objs.values():
            s.clear()

        # reset machines
        for p in self.processes:
            self.machines[p]["busy"] = False
            self.machines[p]["timer"] = 0
            self.machines[p]["timer_init"] = 0
            self.machines[p]["order"] = None

        # orders
        self.pending_orders = []
        for o in self.orders_template:
            job = self.job_types[o["job_type"]]
            self.pending_orders.append(
                Order(
                    order_id=o["id"],
                    job_type=o["job_type"],
                    material=job["material"],
                    route=job["route"],
                    process_times=o.get("process_times", []),
                    arrival_time=o["arrival"],
                    due_time=o["due"],
                )
            )
        self.pending_orders.sort(key=lambda x: x.arrival_time)
        if self.arrival_shift_to_first and self.pending_orders:
            shift = int(self.pending_orders[0].arrival_time) - 1
            if shift > 0:
                for o in self.pending_orders:
                    o.arrival_time = int(o.arrival_time) - shift
                    o.due_time = int(o.due_time) - shift
        self.completed_orders = 0
        self.step_count = 0

    # -------------------
    # Core Gym API
    # -------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_internal_state()
        obs = self._get_obs_dict()
        return obs

    def step(self, action_dict):
        self.step_count += 1
        events = {
            "agv_out_of_bounds": [],
            "agv_wrong_drop": [],
            "agv_delivered": [],
            "agv_picked": [],
            "agv_correct_drop": [],
            "machine_started": [],
            "machine_completed": [],
        }
        boundary_shaping = {}
        distance_shaping = {}
        prev_positions = [p.copy() for p in self.agv_positions]

        # spawn arriving orders
        while self.pending_orders and self.pending_orders[0].arrival_time <= self.step_count:
            order = self.pending_orders.pop(0)
            self.station_objs["SRC"].add_item(order)

        # apply AGV actions
        for i, agv_id in enumerate(self.agv_ids):
            action = int(action_dict.get(agv_id, 0))
            pos = self.agv_positions[i]
            prev = pos.copy()
            carry_before = self.agv_carry[i]
            self.agv_velocities[i][:] = 0.0

            if 1 <= action <= 8:
                # 8-direction move
                delta = self._action_to_delta(action)
                if self._move_by(pos, delta):
                    events["agv_out_of_bounds"].append(agv_id)
                self.agv_velocities[i][:] = (pos - prev).astype(np.float32)

                # Optional dense shaping: reward movement that decreases distance to current target.
                target_pos = self._agv_target_position(i, carry_before)
                if target_pos is not None:
                    dist_prev = float(np.linalg.norm(prev - target_pos))
                    dist_new = float(np.linalg.norm(pos - target_pos))
                    delta = dist_prev - dist_new
                    if self.distance_shaping_mode in {"progress", "pos", "positive"}:
                        delta = max(0.0, delta)
                    distance_shaping[agv_id] = distance_shaping.get(agv_id, 0.0) + float(delta)

            elif self._pick_action_start <= action <= self._pick_action_end:  # pick (ranked index)
                if self.agv_carry[i] is None:
                    station = self._current_station(pos, radius=self.pickup_radius)
                    if station and station != "SINK":
                        pick_rank = int(action - self._pick_action_start)
                        item = self._pop_order_for_pick(self.station_objs[station], pick_rank)
                        if item is not None:
                            self.agv_carry[i] = item
                            events["agv_picked"].append(agv_id)

            elif action == self._drop_action:  # drop
                if self.agv_carry[i] is not None:
                    station = self._current_station(pos, radius=self.drop_radius)
                    if station:
                        if self._is_valid_drop(station, self.agv_carry[i]):
                            if station == "SINK":
                                late = max(0, self.step_count - self.agv_carry[i].due_time)
                                events["agv_delivered"].append({"agent": agv_id, "late_steps": late})
                                self.completed_orders += 1
                                self.agv_carry[i] = None
                            else:
                                self.station_objs[station].add_item(self.agv_carry[i])
                                self.agv_carry[i] = None
                                events["agv_correct_drop"].append(agv_id)
                        else:
                            events["agv_wrong_drop"].append(agv_id)

            if self.boundary_margin > 0.0:
                boundary_shaping[agv_id] = float(self._boundary_shaping(self.agv_positions[i], self.boundary_margin))

        if boundary_shaping:
            events["boundary_shaping"] = boundary_shaping
        if distance_shaping:
            events["distance_shaping"] = distance_shaping

        if self.collision_radius > 0.0 and len(self.agv_ids) >= 2:
            collision_agents = set()
            r2 = float(self.collision_radius) * float(self.collision_radius)
            for i in range(len(self.agv_ids)):
                for j in range(i + 1, len(self.agv_ids)):
                    d = self.agv_positions[i] - self.agv_positions[j]
                    if float(d[0] * d[0] + d[1] * d[1]) <= r2:
                        collision_agents.add(self.agv_ids[i])
                        collision_agents.add(self.agv_ids[j])
            if collision_agents:
                events["collision_agents"] = list(sorted(collision_agents))

        if "collision_agents" in events:
            events["collision_count"] = int(len(events["collision_agents"]))

        if self.global_task_complete_reward:
            completions = int(len(events.get("agv_delivered", [])))
            if completions > 0:
                events["global_task_complete"] = completions

        # machine processing
        machine_selections, machine_events = self._machine_step(action_dict)
        if machine_events:
            events["machine_started"].extend(machine_events.get("machine_started", []))
            events["machine_completed"].extend(machine_events.get("machine_completed", []))

        # reward
        rewards = self.rewarder.compute(self.agent_ids, events)

        # done condition
        done = self.step_count >= self.max_steps or self.completed_orders >= self.total_orders
        dones = {a: done for a in self.agent_ids}
        dones["__all__"] = done

        obs = self._get_obs_dict()
        infos = {
            "completed_orders": self.completed_orders,
            "time": self.step_count,
            "machine_selections": machine_selections,
            "collision_agents": events.get("collision_agents", []),
            "src_queue_len": int(self.station_objs["SRC"].queue_length()),
            "agv_carry_count": int(sum(1 for x in self.agv_carry if x is not None)),
            # per-step event details for diagnostics
            "picked_agents": list(events.get("agv_picked", [])),
            "correct_drop_agents": list(events.get("agv_correct_drop", [])),
            "wrong_drop_agents": list(events.get("agv_wrong_drop", [])),
            "machine_started": list(events.get("machine_started", [])),
            "machine_completed": list(events.get("machine_completed", [])),
        }
        return obs, rewards, dones, infos

    def _agv_target_position(self, agv_index, carry_before):
        # Target for dense shaping:
        # - If carrying: next required station (or SINK if finished)
        # - If not carrying: nearest station (excluding SINK) that currently has at least one Order, else SRC.
        if carry_before is not None:
            if getattr(carry_before, "finished", False):
                target_station = "SINK"
            else:
                next_proc = carry_before.next_process()
                target_station = "SINK" if next_proc is None else self.process_to_station.get(next_proc, "SINK")
            return np.array(self.station_positions[target_station], dtype=np.float32)

        pos = self.agv_positions[agv_index]
        candidates = []
        for s in self.stations:
            if s == "SINK":
                continue
            station_obj = self.station_objs[s]
            station_process = self.station_to_process.get(s, None)
            has_pickable = False
            for item in station_obj.queue:
                if not isinstance(item, Order):
                    continue
                if station_process is not None and (not item.finished) and item.next_process() == station_process:
                    continue
                has_pickable = True
                break
            if has_pickable:
                candidates.append(s)
        if not candidates:
            return np.array(self.station_positions["SRC"], dtype=np.float32)

        best = None
        best_dist = None
        for s in candidates:
            sp = np.array(self.station_positions[s], dtype=np.float32)
            d = float(np.linalg.norm(pos - sp))
            if best is None or d < best_dist:
                best = s
                best_dist = d
        return np.array(self.station_positions[best], dtype=np.float32)

    # -------------------
    # Visualization (matplotlib)
    # -------------------
    def render(self):
        try:
            import matplotlib

            backend = str(matplotlib.get_backend()).lower()
            if backend == "agg":
                try:
                    matplotlib.use("TkAgg", force=True)
                except Exception:
                    pass
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError("matplotlib is required for render(). Install it and retry.") from exc

        if self._viz["fig"] is None:
            plt.ion()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.set_aspect("equal")
            ax.set_xlim(self.bounds["xmin"], self.bounds["xmax"])
            ax.set_ylim(self.bounds["ymin"], self.bounds["ymax"])
            ax.set_title("Factory Env (multi-process)")
            self._viz["fig"] = fig
            self._viz["ax"] = ax
            try:
                fig.show()
            except Exception:
                pass

        fig = self._viz["fig"]
        ax = self._viz["ax"]
        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim(self.bounds["xmin"], self.bounds["xmax"])
        ax.set_ylim(self.bounds["ymin"], self.bounds["ymax"])
        ax.set_title(f"Factory Env (t={self.step_count})")

        # draw stations
        for name, station in self.station_objs.items():
            x, y = station.position
            ax.scatter([x], [y], s=120, marker="s", color="#444444")
            ax.text(x + 0.2, y + 0.2, f"{name}\nq={station.queue_length()}", fontsize=8)

        # draw AGVs
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for i, agv_id in enumerate(self.agv_ids):
            x, y = self.agv_positions[i]
            color = colors[i % len(colors)]
            ax.scatter([x], [y], s=100, marker="o", color=color)
            label = agv_id
            carry = self.agv_carry[i]
            if carry:
                label += f"({carry.job_type})"
            ax.text(x + 0.2, y - 0.2, label, fontsize=8, color=color)

        # machine status
        y0 = self.bounds["ymax"] - 0.5
        for idx, process in enumerate(self.processes):
            machine = self.machines[process]
            status = "busy" if machine["busy"] else "idle"
            ax.text(self.bounds["xmin"] + 0.2, y0 - idx * 0.5, f"{process}: {status}", fontsize=8)

        fig.canvas.draw_idle()
        try:
            fig.canvas.flush_events()
        except Exception:
            pass
        try:
            plt.pause(0.001)
        except Exception:
            pass
        return fig

    def close(self):
        if self._viz.get("fig") is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self._viz["fig"])
            except Exception:
                pass
        self._viz = {"fig": None, "ax": None, "last_step": -1}

    # -------------------
    # Machine logic
    # -------------------
    def _machine_step(self, action_dict=None):
        action_dict = action_dict or {}
        selections = []
        machine_events = {"machine_started": [], "machine_completed": []}

        # start processing if idle
        for process in self.processes:
            machine = self.machines[process]
            if not machine["busy"]:
                station = self.station_objs[self.process_to_station[process]]
                if self.machine_as_agent:
                    mid = self.process_to_machine_id[process]
                    act = int(action_dict.get(mid, 0))
                    # 0 => rank0 (default), 1..K => rank(action-1)
                    pick_rank = max(0, act - 1)
                    order, meta = self._pop_order_for_process_by_rank(station, process, pick_rank, return_meta=True)
                else:
                    order = self._pop_next_for_process(station, process)
                    meta = None

                if order is not None:
                    machine["busy"] = True
                    machine["order"] = order
                    default_t = self.process_time_defaults.get(process, 10)
                    machine["timer"] = order.current_process_time(default_t)
                    machine["timer_init"] = int(machine["timer"])
                    if self.machine_as_agent:
                        selections.append(
                            {
                                "machine_id": self.process_to_machine_id[process],
                                "process": process,
                                "action": int(meta.get("action", 0) if meta else 0),
                                "pick_rank": int(meta.get("pick_rank", 0) if meta else 0),
                                "eligible_count": int(meta.get("eligible_count", 0) if meta else 0),
                                "chosen_order_id": int(order.order_id),
                            }
                        )
                    machine_events["machine_started"].append(self.process_to_machine_id[process])

        # progress processing
        for process in self.processes:
            machine = self.machines[process]
            if machine["busy"]:
                machine["timer"] -= 1
                if machine["timer"] <= 0:
                    order = machine["order"]
                    order.advance()
                    station = self.station_objs[self.process_to_station[process]]
                    station.add_item(order)
                    machine["busy"] = False
                    machine["timer"] = 0
                    machine["timer_init"] = 0
                    machine["order"] = None
                    machine_events["machine_completed"].append(self.process_to_machine_id[process])
        return selections, machine_events

    def _pop_next_for_process(self, station, process):
        for idx, item in enumerate(station.queue):
            if isinstance(item, Order) and item.next_process() == process:
                return station.queue.pop(idx)
        return None

    def _pop_order_for_process_by_rank(self, station, process, pick_rank, return_meta=False):
        candidates = [(idx, item) for idx, item in enumerate(station.queue) if isinstance(item, Order) and item.next_process() == process]
        if not candidates:
            if return_meta:
                return None, {"action": 0, "pick_rank": int(pick_rank), "eligible_count": 0}
            return None

        if self.machine_sort_key == "fifo":
            ranked = candidates
        elif self.machine_sort_key in {"earliest_due", "edd"}:
            ranked = sorted(candidates, key=lambda x: (x[1].due_time, x[1].arrival_time, x[1].order_id))
        elif self.machine_sort_key in {"highest_priority", "priority"}:
            ranked = sorted(candidates, key=lambda x: (-int(getattr(x[1], "priority", 0)), x[1].due_time, x[1].order_id))
        elif self.machine_sort_key in {"priority_then_due", "priority_due"}:
            ranked = sorted(
                candidates,
                key=lambda x: (-int(getattr(x[1], "priority", 0)), x[1].due_time, x[1].arrival_time, x[1].order_id),
            )
        else:
            ranked = candidates

        pick_rank = int(max(0, pick_rank))
        pick_rank = min(pick_rank, len(ranked) - 1)
        remove_idx, _ = ranked[pick_rank]
        order = station.queue.pop(remove_idx)
        if return_meta:
            return order, {"action": int(pick_rank + 1), "pick_rank": int(pick_rank), "eligible_count": int(len(ranked))}
        return order

    def get_action_mask(self, agent_id):
        if agent_id not in self.action_spaces:
            return None
        n = int(self.action_spaces[agent_id].n)
        mask = np.ones((n,), dtype=np.float32)

        # AGV masks
        if agent_id in self.agv_ids:
            i = self.agv_ids.index(agent_id)
            pos = self.agv_positions[i]
            carry = self.agv_carry[i]

            # Prevent moves that would be clamped out-of-bounds.
            for a in range(1, 9):
                delta = self._action_to_delta(a)
                d = np.array(delta, dtype=np.float32)
                norm = float(np.linalg.norm(d))
                if norm < 1e-6:
                    mask[a] = 0.0
                    continue
                proposed = pos + (d / norm) * float(self.agv_speed)
                clamped = proposed.copy()
                clamped[0] = min(max(clamped[0], self.bounds["xmin"]), self.bounds["xmax"])
                clamped[1] = min(max(clamped[1], self.bounds["ymin"]), self.bounds["ymax"])
                if not np.allclose(proposed, clamped):
                    mask[a] = 0.0

            # Pick actions
            if carry is not None:
                mask[self._pick_action_start : self._pick_action_end + 1] = 0.0
            else:
                station = self._current_station(pos, radius=self.pickup_radius)
                if (station is None) or (station == "SINK"):
                    mask[self._pick_action_start : self._pick_action_end + 1] = 0.0
                else:
                    orders = [item for item in self.station_objs[station].queue if isinstance(item, Order)]
                    if len(orders) == 0:
                        mask[self._pick_action_start : self._pick_action_end + 1] = 0.0
                    station_process = self.station_to_process.get(station, None)
                    if station_process is not None:
                        orders = [o for o in orders if (o.finished or o.next_process() != station_process)]
                        if len(orders) == 0:
                            mask[self._pick_action_start : self._pick_action_end + 1] = 0.0
                    for a in range(self._pick_action_start, self._pick_action_end + 1):
                        rank = int(a - self._pick_action_start)
                        if rank >= len(orders):
                            mask[a] = 0.0

            # Drop action
            if carry is None:
                mask[self._drop_action] = 0.0
            else:
                station = self._current_station(pos, radius=self.drop_radius)
                if station is None:
                    mask[self._drop_action] = 0.0
                else:
                    # Enforce valid routing: only allow dropping at the correct station (or SINK when finished).
                    if not self._is_valid_drop(station, carry):
                        mask[self._drop_action] = 0.0

            return mask

        # Machine masks
        if self.machine_as_agent and agent_id in self.machine_ids:
            process = str(agent_id).replace("machine_", "", 1)
            machine = self.machines.get(process)
            if machine is None:
                return mask
            if machine["busy"]:
                mask[:] = 0.0
                mask[0] = 1.0
                return mask

            station = self.station_objs[self.process_to_station[process]]
            eligible = [item for item in station.queue if isinstance(item, Order) and item.next_process() == process]
            if len(eligible) == 0:
                mask[:] = 0.0
                mask[0] = 1.0
            return mask

        return mask

    def get_action_masks(self):
        masks = {}
        for a in self.agent_ids:
            m = self.get_action_mask(a)
            if m is not None:
                masks[a] = m
        return masks

    def _pop_order_for_pick(self, station, pick_rank):
        station_process = self.station_to_process.get(station.name, None)
        candidates = []
        for idx, item in enumerate(station.queue):
            if not isinstance(item, Order):
                continue
            # Don't let AGVs pick jobs that are waiting to be processed at this station.
            if station_process is not None and (not item.finished) and item.next_process() == station_process:
                continue
            candidates.append((idx, item))
        if not candidates:
            return None

        if self.pick_sort_key == "fifo":
            ranked = candidates
        elif self.pick_sort_key in {"earliest_due", "edd"}:
            ranked = sorted(candidates, key=lambda x: (x[1].due_time, x[1].arrival_time, x[1].order_id))
        elif self.pick_sort_key in {"highest_priority", "priority"}:
            ranked = sorted(candidates, key=lambda x: (-int(getattr(x[1], "priority", 0)), x[1].due_time, x[1].order_id))
        elif self.pick_sort_key in {"priority_then_due", "priority_due"}:
            ranked = sorted(
                candidates,
                key=lambda x: (-int(getattr(x[1], "priority", 0)), x[1].due_time, x[1].arrival_time, x[1].order_id),
            )
        else:
            ranked = candidates

        if pick_rank < 0 or pick_rank >= len(ranked):
            return None
        remove_idx, _ = ranked[pick_rank]
        return station.queue.pop(remove_idx)

    # -------------------
    # Helper functions
    # -------------------
    def _move_towards(self, pos, target):
        direction = target - pos
        dist = np.linalg.norm(direction)
        if dist > 1e-6:
            step = (direction / dist) * self.agv_speed
            if np.linalg.norm(step) > dist:
                step = direction
            proposed = pos + step
        else:
            proposed = pos.copy()

        clamped = proposed.copy()
        clamped[0] = min(max(clamped[0], self.bounds["xmin"]), self.bounds["xmax"])
        clamped[1] = min(max(clamped[1], self.bounds["ymin"]), self.bounds["ymax"])

        out_of_bounds = not np.allclose(proposed, clamped)
        pos[:] = clamped
        return out_of_bounds

    def _action_to_delta(self, action):
        # 1..8 => N,S,W,E,NW,NE,SW,SE
        if action == 1:
            return np.array([0.0, 1.0], dtype=np.float32)
        if action == 2:
            return np.array([0.0, -1.0], dtype=np.float32)
        if action == 3:
            return np.array([-1.0, 0.0], dtype=np.float32)
        if action == 4:
            return np.array([1.0, 0.0], dtype=np.float32)
        if action == 5:
            return np.array([-1.0, 1.0], dtype=np.float32)
        if action == 6:
            return np.array([1.0, 1.0], dtype=np.float32)
        if action == 7:
            return np.array([-1.0, -1.0], dtype=np.float32)
        return np.array([1.0, -1.0], dtype=np.float32)

    def _move_by(self, pos, direction):
        direction = np.array(direction, dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return False
        proposed = pos + (direction / norm) * float(self.agv_speed)

        clamped = proposed.copy()
        clamped[0] = min(max(clamped[0], self.bounds["xmin"]), self.bounds["xmax"])
        clamped[1] = min(max(clamped[1], self.bounds["ymin"]), self.bounds["ymax"])

        out_of_bounds = not np.allclose(proposed, clamped)
        pos[:] = clamped
        return out_of_bounds

    def _boundary_shaping(self, pos, margin):
        margin = float(margin)
        if margin <= 0.0:
            return 0.0
        xmin, ymin, xmax, ymax = self.bounds["xmin"], self.bounds["ymin"], self.bounds["xmax"], self.bounds["ymax"]
        d_left = float(pos[0] - xmin)
        d_right = float(xmax - pos[0])
        d_bottom = float(pos[1] - ymin)
        d_top = float(ymax - pos[1])
        min_dist = min(d_left, d_right, d_bottom, d_top)
        if min_dist >= margin:
            return 0.0
        risk = (margin - min_dist) / margin  # 0..1
        return -float(risk)

    def _near(self, pos, target, radius):
        return np.linalg.norm(pos - target) <= radius

    def _current_station(self, pos, radius):
        for name, station in self.station_objs.items():
            if self._near(pos, np.array(station.position, dtype=np.float32), radius=radius):
                return name
        return None

    def _is_valid_drop(self, station, order):
        if order.finished:
            return station == "SINK"
        next_proc = order.next_process()
        if next_proc is None:
            return station == "SINK"
        return station == self.process_to_station.get(next_proc)

    def _one_hot(self, items, value):
        vec = np.zeros(len(items), dtype=np.float32)
        if value in items:
            vec[items.index(value)] = 1.0
        return vec

    def _get_obs_dict(self):
        obs = {}
        queue_lens = [float(self.station_objs[s].queue_length()) for s in self.stations]
        machine_busy = [1.0 if self.machines[p]["busy"] else 0.0 for p in self.processes]
        machine_timer = None
        if self.obs_include_machine_timer:
            machine_timer = [
                (float(self.machines[p]["timer"]) / float(max(1, int(self.machines[p].get("timer_init", 0)))))
                if self.machines[p]["busy"]
                else 0.0
                for p in self.processes
            ]
        time_frac = float(self.step_count) / float(max(1, self.max_steps))

        job_keys = list(self.job_types.keys())
        for i, agv_id in enumerate(self.agv_ids):
            pos = self.agv_positions[i].astype(np.float32)
            vel = self.agv_velocities[i].astype(np.float32)

            station_pos_feat = None
            if self.obs_include_station_positions:
                xmin, ymin, xmax, ymax = self.bounds["xmin"], self.bounds["ymin"], self.bounds["xmax"], self.bounds["ymax"]
                span_x = float(max(1e-6, xmax - xmin))
                span_y = float(max(1e-6, ymax - ymin))
                feats = []
                for s in self.stations:
                    sx, sy = self.station_positions[s]
                    if self.obs_station_pos_mode in {"absolute", "abs"}:
                        x, y = float(sx), float(sy)
                        if self.obs_station_pos_normalize:
                            x = (x - float(xmin)) / span_x
                            y = (y - float(ymin)) / span_y
                        feats.extend([x, y])
                    else:
                        dx = float(sx) - float(pos[0])
                        dy = float(sy) - float(pos[1])
                        if self.obs_station_pos_normalize:
                            dx = dx / span_x
                            dy = dy / span_y
                        feats.extend([dx, dy])
                station_pos_feat = np.array(feats, dtype=np.float32)

            neighbor_feat = None
            if self.obs_include_neighbor_agvs and int(self.obs_neighbor_k) > 0:
                xmin, ymin, xmax, ymax = self.bounds["xmin"], self.bounds["ymin"], self.bounds["xmax"], self.bounds["ymax"]
                span_x = float(max(1e-6, xmax - xmin))
                span_y = float(max(1e-6, ymax - ymin))

                others = []
                for j, other_id in enumerate(self.agv_ids):
                    if j == i:
                        continue
                    other_pos = self.agv_positions[j]
                    dx = float(other_pos[0] - pos[0])
                    dy = float(other_pos[1] - pos[1])
                    dist2 = dx * dx + dy * dy
                    others.append((dist2, dx, dy))
                others.sort(key=lambda x: x[0])

                k = int(self.obs_neighbor_k)
                feats = np.zeros(2 * k, dtype=np.float32)
                for idx in range(min(k, len(others))):
                    _, dx, dy = others[idx]
                    if self.obs_neighbor_normalize:
                        dx = dx / span_x
                        dy = dy / span_y
                    feats[2 * idx] = float(dx)
                    feats[2 * idx + 1] = float(dy)
                neighbor_feat = feats

            target_feat = None
            if self.obs_include_target_vector:
                xmin, ymin, xmax, ymax = self.bounds["xmin"], self.bounds["ymin"], self.bounds["xmax"], self.bounds["ymax"]
                span_x = float(max(1e-6, xmax - xmin))
                span_y = float(max(1e-6, ymax - ymin))
                target_pos = self._agv_target_position(i, self.agv_carry[i])
                if target_pos is None:
                    target_feat = np.zeros(2, dtype=np.float32)
                else:
                    dx = float(target_pos[0] - pos[0])
                    dy = float(target_pos[1] - pos[1])
                    if self.obs_target_normalize:
                        dx = dx / span_x
                        dy = dy / span_y
                    target_feat = np.array([dx, dy], dtype=np.float32)
            carry = self.agv_carry[i]
            carry_flag = 0.0 if carry is None else 1.0
            carry_job = self._one_hot(job_keys, carry.job_type) if carry else np.zeros(len(job_keys), dtype=np.float32)
            carry_next = (
                self._one_hot(self.processes, carry.next_process())
                if carry
                else np.zeros(len(self.processes), dtype=np.float32)
            )
 
            current_station = self._current_station(pos, radius=self.obs_station_radius)
            cur_station_oh = (
                self._one_hot(self.stations, current_station)
                if (self.obs_include_current_station and current_station is not None)
                else (np.zeros(len(self.stations), dtype=np.float32) if self.obs_include_current_station else None)
            )

            pick_window = None
            if self.obs_pick_window_k > 0:
                pick_window = np.zeros(
                    int(self.obs_pick_window_k) * (1 + 1 + len(self.processes)), dtype=np.float32
                )
                if current_station is not None and current_station in self.station_objs:
                    station_obj = self.station_objs[current_station]
                    station_process = self.station_to_process.get(current_station, None)
                    candidates = []
                    for item in station_obj.queue:
                        if not isinstance(item, Order):
                            continue
                        if station_process is not None and (not item.finished) and item.next_process() == station_process:
                            continue
                        candidates.append(item)
                    if candidates:
                        if self.pick_sort_key == "fifo":
                            ranked = candidates
                        elif self.pick_sort_key in {"earliest_due", "edd"}:
                            ranked = sorted(candidates, key=lambda o: (o.due_time, o.arrival_time, o.order_id))
                        elif self.pick_sort_key in {"highest_priority", "priority"}:
                            ranked = sorted(candidates, key=lambda o: (-int(getattr(o, "priority", 0)), o.due_time, o.order_id))
                        elif self.pick_sort_key in {"priority_then_due", "priority_due"}:
                            ranked = sorted(
                                candidates,
                                key=lambda o: (-int(getattr(o, "priority", 0)), o.due_time, o.arrival_time, o.order_id),
                            )
                        else:
                            ranked = candidates

                        for j in range(min(int(self.obs_pick_window_k), len(ranked))):
                            o = ranked[j]
                            due_remaining = float(o.due_time - self.step_count) / float(max(1, self.max_steps))
                            priority = float(getattr(o, "priority", 0))
                            next_proc = o.next_process()
                            next_oh = self._one_hot(self.processes, next_proc) if next_proc else np.zeros(
                                len(self.processes), dtype=np.float32
                            )
                            start = j * (1 + 1 + len(self.processes))
                            pick_window[start] = due_remaining
                            pick_window[start + 1] = priority
                            pick_window[start + 2 : start + 2 + len(self.processes)] = next_oh
            obs_vec = np.concatenate(
                [
                    pos,
                    vel,
                    *( [station_pos_feat] if station_pos_feat is not None else [] ),
                    *( [neighbor_feat] if neighbor_feat is not None else [] ),
                    *( [target_feat] if target_feat is not None else [] ),
                    [carry_flag],
                    carry_job,
                    carry_next,
                    np.array(queue_lens, dtype=np.float32),
                    np.array(machine_busy, dtype=np.float32),
                    *( [np.array(machine_timer, dtype=np.float32)] if machine_timer is not None else [] ),
                    [time_frac],
                    *( [cur_station_oh] if cur_station_oh is not None else [] ),
                    *( [pick_window] if pick_window is not None else [] ),
                ]
            ).astype(np.float32)
            obs[agv_id] = obs_vec

        if self.machine_as_agent:
            proc_denom = float(max(1, int(getattr(self, "process_time_max", 1))))
            for process in self.processes:
                mid = self.process_to_machine_id[process]
                machine = self.machines[process]
                busy = 1.0 if machine["busy"] else 0.0
                timer_frac = (
                    float(machine["timer"]) / float(max(1, int(machine.get("timer_init", 0)))) if machine["busy"] else 0.0
                )

                station = self.station_objs[self.process_to_station[process]]
                eligible = [item for item in station.queue if isinstance(item, Order) and item.next_process() == process]
                eligible_q_len = float(len(eligible))

                topk = np.zeros(int(self.machine_obs_k) * (1 + 1 + 1), dtype=np.float32)
                if eligible:
                    if self.machine_sort_key == "fifo":
                        ranked = eligible
                    elif self.machine_sort_key in {"earliest_due", "edd"}:
                        ranked = sorted(eligible, key=lambda o: (o.due_time, o.arrival_time, o.order_id))
                    elif self.machine_sort_key in {"highest_priority", "priority"}:
                        ranked = sorted(eligible, key=lambda o: (-int(getattr(o, "priority", 0)), o.due_time, o.order_id))
                    elif self.machine_sort_key in {"priority_then_due", "priority_due"}:
                        ranked = sorted(
                            eligible,
                            key=lambda o: (-int(getattr(o, "priority", 0)), o.due_time, o.arrival_time, o.order_id),
                        )
                    else:
                        ranked = eligible

                    default_t = int(self.process_time_defaults.get(process, 10))
                    for j in range(min(int(self.machine_obs_k), len(ranked))):
                        o = ranked[j]
                        due_remaining = float(o.due_time - self.step_count) / float(max(1, self.max_steps))
                        priority = float(getattr(o, "priority", 0))
                        pt = float(o.current_process_time(default_t)) / proc_denom
                        start = j * 3
                        topk[start] = due_remaining
                        topk[start + 1] = priority
                        topk[start + 2] = pt

                obs[mid] = np.concatenate(
                    [[busy], [timer_frac], [eligible_q_len], [time_frac], topk]
                ).astype(np.float32)
        return obs

    # expose action/obs spaces for external trainer
    def get_action_space(self, agent_id):
        return self.action_spaces[agent_id]

    def get_observation_space(self, agent_id):
        return self.observation_spaces[agent_id]
