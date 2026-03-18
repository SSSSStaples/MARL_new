import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
from typing import Optional

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
      rewards: dict agent_id -> float
      dones: dict agent_id -> bool, plus '__all__' key
      infos: dict with summary fields
    """

    metadata = {"render_modes": []}

    def   __init__(self, config_path="configs/default.yaml"):
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

        # env params
        self.max_steps = int(env_cfg.get("max_steps", 300))
        self.agv_count = int(env_cfg.get("agv_count", 2))
        self.agv_speed = float(env_cfg.get("agv_speed", 0.2))
        self.pickup_radius = float(env_cfg.get("pickup_radius", 0.5))
        self.drop_radius = float(env_cfg.get("drop_radius", 0.5))
        self.render_pause_s = float(env_cfg.get("render_pause_s", 0.001))

        # Optional: soft boundary shaping (discourage hugging walls).
        # - boundary_margin<=0 disables this shaping (default).
        self.boundary_margin = float(env_cfg.get("boundary_margin", 0.0))

        # Optional: allow agents to choose which queued order to pick.
        # - pickup_index_count=1 keeps the original (single "pick" action).
        # - pickup_index_count>1 expands the action space so that "pick" becomes K actions
        #   selecting the 0..K-1 ranked order within the current station queue.
        self.pickup_index_count = int(env_cfg.get("pickup_index_count", 1))
        self.pick_sort_key = str(env_cfg.get("pick_sort_key", "fifo")).strip().lower()

        # Optional: add scheduling-relevant features to observations.
        # - obs_include_current_station: append one-hot of current station (within obs_station_radius).
        # - obs_pick_window_k: append top-K order features at current station (ranked by pick_sort_key).
        obs_cfg = env_cfg.get("obs", {}) or {}
        self.obs_station_radius = float(obs_cfg.get("station_radius", max(self.pickup_radius, self.drop_radius)))
        self.obs_include_current_station = bool(obs_cfg.get("include_current_station", False))
        self.obs_pick_window_k = int(obs_cfg.get("pick_window_k", 0))
        self.obs_include_machine_timer = bool(obs_cfg.get("include_machine_timer", False))
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

        # reward calculator
        self.rewarder = RewardCalculator(reward_cfg)

        # agents list
        self.agent_ids = [f"agv_{i+1}" for i in range(self.agv_count)]

        # action spaces (per agent)
        # 0 idle
        # 1..8 move (N,S,W,E,NW,NE,SW,SE) with step size agv_speed
        # pick actions: 9..(9+pickup_index_count-1), drop: 9+pickup_index_count
        self._pick_action_start = 9
        self._pick_action_end = self._pick_action_start + max(1, self.pickup_index_count) - 1
        self._drop_action = self._pick_action_end + 1
        self.action_spaces = {a: spaces.Discrete(self._drop_action + 1) for a in self.agent_ids}

        # observation spaces
        # [pos(2), vel(2), carry_flag(1), carry_job(onehot), carry_next_process(onehot),
        #  queue_lens(len(stations)), machine_busy(len(processes)), (optional) machine_timer(len(processes)), time_frac(1)]
        obs_dim = (
            2
            + 2
            + 1
            + len(self.job_types)
            + len(self.processes)
            + len(self.stations)
            + len(self.processes)
            + 1
        )
        if self.obs_include_machine_timer:
            obs_dim += len(self.processes)
        if self.obs_include_current_station:
            obs_dim += len(self.stations)
        if self.obs_pick_window_k > 0:
            per_order_dim = 1 + 1 + len(self.processes)  # due_remaining, priority, next_process onehot
            obs_dim += int(self.obs_pick_window_k) * int(per_order_dim)
        self.observation_spaces = {
            a: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32) for a in self.agent_ids
        }

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
        self.agv_positions = [np.array(self.station_positions["SRC"], dtype=np.float32) for _ in self.agent_ids]
        self.agv_velocities = [np.zeros(2, dtype=np.float32) for _ in self.agent_ids]
        self.agv_carry: list[Optional[Order]] = [None for _ in self.agent_ids]

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
        events = {"agv_out_of_bounds": [], "agv_wrong_drop": [], "agv_delivered": []}
        boundary_shaping = {}

        # spawn arriving orders
        while self.pending_orders and self.pending_orders[0].arrival_time <= self.step_count:
            order = self.pending_orders.pop(0)
            self.station_objs["SRC"].add_item(order)

        # apply AGV actions
        for i, agv_id in enumerate(self.agent_ids):
            action = int(action_dict.get(agv_id, 0))
            pos = self.agv_positions[i]
            prev = pos.copy()
            self.agv_velocities[i][:] = 0.0

            if 1 <= action <= 8:
                # 8-direction move
                delta = self._action_to_delta(action)
                if self._move_by(pos, delta):
                    events["agv_out_of_bounds"].append(agv_id)
                self.agv_velocities[i][:] = (pos - prev).astype(np.float32)

            elif self._pick_action_start <= action <= self._pick_action_end:  # pick (ranked index)
                if self.agv_carry[i] is None:
                    station = self._current_station(pos, radius=self.pickup_radius)
                    if station and station != "SINK":
                        pick_rank = int(action - self._pick_action_start)
                        item = self._pop_order_for_pick(self.station_objs[station], pick_rank)
                        if item is not None:
                            self.agv_carry[i] = item

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
                        else:
                            events["agv_wrong_drop"].append(agv_id)

            if self.boundary_margin > 0.0:
                boundary_shaping[agv_id] = float(self._boundary_shaping(self.agv_positions[i], self.boundary_margin))

        if boundary_shaping:
            events["boundary_shaping"] = boundary_shaping

        # machine processing
        self._machine_step()

        # reward
        rewards = self.rewarder.compute(self.agent_ids, events)

        # done condition
        done = self.step_count >= self.max_steps or self.completed_orders >= self.total_orders
        dones = {a: done for a in self.agent_ids}
        dones["__all__"] = done

        obs = self._get_obs_dict()
        infos = {"completed_orders": self.completed_orders, "time": self.step_count}
        return obs, rewards, dones, infos

    # -------------------
    # Machine logic
    # -------------------
    def _machine_step(self):
        # start processing if idle
        for process in self.processes:
            machine = self.machines[process]
            if not machine["busy"]:
                station = self.station_objs[self.process_to_station[process]]
                order = self._pop_next_for_process(station, process)
                if order is not None:
                    machine["busy"] = True
                    machine["order"] = order
                    default_t = self.process_time_defaults.get(process, 10)
                    machine["timer"] = order.current_process_time(default_t)
                    machine["timer_init"] = int(machine["timer"])

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

    def _pop_next_for_process(self, station, process):
        for idx, item in enumerate(station.queue):
            if isinstance(item, Order) and item.next_process() == process:
                return station.queue.pop(idx)
        return None

    def _pop_order_for_pick(self, station, pick_rank):
        candidates = [(idx, item) for idx, item in enumerate(station.queue) if isinstance(item, Order)]
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
            # Normalize by the current job's duration so values are always in [0, 1].
            machine_timer = [
                (float(self.machines[p]["timer"]) / float(max(1, int(self.machines[p].get("timer_init", 0)))))
                if self.machines[p]["busy"]
                else 0.0
                for p in self.processes
            ]
        time_frac = float(self.step_count) / float(max(1, self.max_steps))

        job_keys = list(self.job_types.keys())
        for i, agv_id in enumerate(self.agent_ids):
            pos = self.agv_positions[i].astype(np.float32)
            vel = self.agv_velocities[i].astype(np.float32)
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
                    candidates = [item for item in station_obj.queue if isinstance(item, Order)]
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
        return obs

    # expose action/obs spaces for external trainer
    def get_action_space(self, agent_id):
        return self.action_spaces[agent_id]

    def get_observation_space(self, agent_id):
        return self.observation_spaces[agent_id]

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
        for i, agv_id in enumerate(self.agent_ids):
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
            plt.pause(float(self.render_pause_s))
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
