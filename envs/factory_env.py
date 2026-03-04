import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
from .station_objects import Station, Material, Product


class FactoryEnv(gym.Env):
    """
    Multi-agent Factory Environment.

    Agents:
      - 'mover_1' : Team-1, X -> Y transport (materials)
      - 'mover_2' : Team-2, Y -> Z transport (products/components)
      - 'manuf_1' : Manufacturing machine located at Station Y

    Multi-agent API:
      obs, rewards, dones, infos = env.step(action_dict)
      obs: dict agent_id -> np.array
      rewards: dict agent_id -> float
      dones: dict agent_id -> bool, plus '__all__' key
      infos: dict agent_id -> {}
    """

    metadata = {"render_modes": []}

    def __init__(self, config_path="configs/default.yaml"):
        super().__init__()
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        env_cfg = cfg["env"]
        reward_cfg = cfg["reward"]

        # env params
        self.max_steps = int(env_cfg.get("max_steps", 200))
        self.station_positions = env_cfg.get("station_positions", {"X": [1.0, 5.0], "Y": [5.0, 5.0], "Z": [9.0, 5.0]})
        self.mover_speed = float(env_cfg.get("mover_speed", 0.2))
        self.pickup_radius = float(env_cfg.get("pickup_radius", 0.5))
        self.drop_radius = float(env_cfg.get("drop_radius", 0.5))
        self.manufacturing_time_required = int(env_cfg.get("manufacturing_time", 5))

        # reward params
        self.r_task = float(reward_cfg.get("task_complete", 50.0))
        self.r_time = float(reward_cfg.get("time_penalty", -0.1))
        self.r_collision = float(reward_cfg.get("collision_penalty", -10.0))
        self.r_distance = float(reward_cfg.get("distance_shaping", 0.5))

        # Stations
        self.station_X = Station("X", self.station_positions["X"])
        self.station_Y = Station("Y", self.station_positions["Y"])
        self.station_Z = Station("Z", self.station_positions["Z"])

        # Agents list
        self.agent_ids = ["mover_1", "mover_2", "manuf_1"]

        # Per-agent action spaces (discrete)
        # mover_1: 0 idle,1 move_to_X,2 move_to_Y,3 pick,4 drop
        # mover_2: 0 idle,1 move_to_Y,2 move_to_Z,3 pick,4 drop
        # manuf_1: 0 idle,1 request_material (set need),2 start_if_has_material
        self.action_spaces = {
            "mover_1": spaces.Discrete(5),
            "mover_2": spaces.Discrete(5),
            "manuf_1": spaces.Discrete(3),
        }

        # Observations: per-agent small vectors
        # mover obs: pos(x,y), carry_flag(0/1), carry_type(one-hot 3), station queue lens (X,Y,Z), manuf_need, manuf_busy
        # manuf obs: machine_busy, machine_timer, queue_Y_len, next_required_type(one-hot maybe), recent_throughput
        mover_obs_dim = 2 + 1 + 3 + 3 + 1 + 1  # pos(2), carry_flag, carry type one-hot, queues(3), manuf_need, manuf_busy
        manuf_obs_dim = 1 + 1 + 1 + 3 + 1  # busy, timer, queueYlen, required_type(one-hot 3), manuf_need flag (redundant but okay)

        self.observation_spaces = {
            "mover_1": spaces.Box(low=-np.inf, high=np.inf, shape=(mover_obs_dim,), dtype=np.float32),
            "mover_2": spaces.Box(low=-np.inf, high=np.inf, shape=(mover_obs_dim,), dtype=np.float32),
            "manuf_1": spaces.Box(low=-np.inf, high=np.inf, shape=(manuf_obs_dim,), dtype=np.float32),
        }

        # internal state
        self._init_internal_state()

    def _init_internal_state(self):
        # agent positions
        self.mover1_pos = np.array(self.station_positions["X"], dtype=np.float32)
        self.mover2_pos = np.array(self.station_positions["Y"], dtype=np.float32)
        # carries
        self.mover1_carry = None  # Material or Product
        self.mover2_carry = None
        # manufacturing
        self.machine_busy = False
        self.machine_timer = 0
        self.machine_need_material = False  # machine requests material when it needs input
        self.machine_current_material = None

        # counters
        self.step_count = 0
        self.total_throughput = 0  # finished products delivered to Z

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # reset stations
        self.station_X.queue = []
        self.station_Y.queue = []
        self.station_Z.queue = []

        # seed initial materials at X (random types)
        for _ in range(4):
            t = np.random.choice(["shaft", "flange", "plate"])
            self.station_X.add_item(Material(t))

        # reset internal states
        self._init_internal_state()
        # initially machine needs material if Y queue empty
        self.machine_need_material = len(self.station_Y.queue) == 0

        # build obs dict
        obs = self._get_obs_dict()
        return obs

    def step(self, action_dict):
        """
        action_dict: {agent_id: action_int}
        returns: obs_dict, reward_dict, done_dict, info_dict
        """

        self.step_count += 1
        rewards = {a: 0.0 for a in self.agent_ids}
        infos = {a: {} for a in self.agent_ids}

        # --- Apply mover_1 action ---
        a1 = int(action_dict.get("mover_1", 0))
        # mover1 handles X <-> Y
        if a1 == 1:
            self._move_towards(self.mover1_pos, np.array(self.station_positions["X"]))
        elif a1 == 2:
            self._move_towards(self.mover1_pos, np.array(self.station_positions["Y"]))
        elif a1 == 3:  # pick at X
            if self._near(self.mover1_pos, np.array(self.station_positions["X"])):
                if self.mover1_carry is None:
                    mat = self.station_X.remove_item()
                    if mat is not None:
                        self.mover1_carry = mat
        elif a1 == 4:  # drop at Y
            if self._near(self.mover1_pos, np.array(self.station_positions["Y"])):
                if self.mover1_carry is not None:
                    self.station_Y.add_item(self.mover1_carry)
                    # if machine needs, satisfy it by marking need flag false (machine will pick later)
                    self.mover1_carry = None
                    rewards["mover_1"] += self.r_task * 0.5  # partial reward for delivering to Y

        # shaping: reward closer to target
        # if mover1 moving to Y give small shaping based on distance to Y when action 2
        if a1 == 2:
            dist = np.linalg.norm(self.mover1_pos - np.array(self.station_positions["Y"]))
            rewards["mover_1"] += max(0, (1.0 - dist / (np.linalg.norm(np.array(self.station_positions["Y"]) - np.array(self.station_positions["X"]))))) * self.r_distance * 0.01

        # --- Apply mover_2 action ---
        a2 = int(action_dict.get("mover_2", 0))
        # mover2 handles Y <-> Z
        if a2 == 1:
            self._move_towards(self.mover2_pos, np.array(self.station_positions["Y"]))
        elif a2 == 2:
            self._move_towards(self.mover2_pos, np.array(self.station_positions["Z"]))
        elif a2 == 3:  # pick at Y
            if self._near(self.mover2_pos, np.array(self.station_positions["Y"])):
                if self.mover2_carry is None:
                    item = self.station_Y.remove_item()
                    if item is not None:
                        self.mover2_carry = item
        elif a2 == 4:  # drop at Z
            if self._near(self.mover2_pos, np.array(self.station_positions["Z"])):
                if self.mover2_carry is not None:
                    # delivering to Z means finished product accepted (we'll count throughput if product)
                    if isinstance(self.mover2_carry, Product):
                        self.total_throughput += 1
                        rewards["mover_2"] += self.r_task
                        # global reward will also be added below
                        self.station_Z.add_item(self.mover2_carry)
                        self.mover2_carry = None
                    else:
                        # dropping raw material at Z (not ideal) - small negative
                        self.station_Z.add_item(self.mover2_carry)
                        self.mover2_carry = None
                        rewards["mover_2"] += -1.0

        # small shaping reward
        if a2 == 2:
            dist = np.linalg.norm(self.mover2_pos - np.array(self.station_positions["Z"]))
            rewards["mover_2"] += max(0, (1.0 - dist / (np.linalg.norm(np.array(self.station_positions["Z"]) - np.array(self.station_positions["Y"]))))) * self.r_distance * 0.01

        # --- Apply manuf_1 action ---
        a3 = int(action_dict.get("manuf_1", 0))
        # manuf actions: 0 idle,1 request_material,2 start_if_has_material
        if a3 == 1:
            # machine signals need for material
            if (not self.machine_busy) and len(self.station_Y.queue) == 0:
                self.machine_need_material = True
        elif a3 == 2:
            # start processing if material available
            if (not self.machine_busy) and len(self.station_Y.queue) > 0:
                mat = self.station_Y.remove_item()
                if mat is not None:
                    self.machine_current_material = mat
                    self.machine_busy = True
                    self.machine_timer = self.manufacturing_time_required
                    self.machine_need_material = False
                    # small local reward for starting processing
                    rewards["manuf_1"] += 1.0

        # --- Manufacturing progression (time-based) ---
        if self.machine_busy:
            self.machine_timer -= 1
            if self.machine_timer <= 0:
                # finish and push product into Y (finished product sits at Y until mover_2 picks up)
                prod = Product(self.machine_current_material.type)
                self.station_Y.add_item(prod)
                self.machine_busy = False
                self.machine_current_material = None
                # when product produced, small reward assigned (global throughput counted when delivered to Z)
                rewards["manuf_1"] += 2.0

        # --- time penalty applied to all agents ---
        for a in self.agent_ids:
            rewards[a] += self.r_time

        # --- global reward (throughput) shaping: small shared reward when product delivered to Z this step ---
        global_reward = 0.0
        # if total_throughput increased this step, give a global boost
        # (we didn't track previous value, so simplest: if reward for mover2 included r_task we already added big reward)
        # we keep global_reward=0 for now or could implement difference rewards in future

        # --- done condition ---
        done = self.step_count >= self.max_steps
        dones = {a: done for a in self.agent_ids}
        dones["__all__"] = done

        obs = self._get_obs_dict()

        infos.update({"global_throughput": self.total_throughput})

        return obs, rewards, dones, infos

    # -------------------
    # Helper functions
    # -------------------
    def _move_towards(self, pos, target):
        direction = target - pos
        dist = np.linalg.norm(direction)
        if dist > 1e-6:
            step = (direction / dist) * self.mover_speed
            # do not overshoot
            if np.linalg.norm(step) > dist:
                step = direction
            pos += step

    def _near(self, pos, target):
        return np.linalg.norm(pos - target) <= self.pickup_radius

    def _one_hot_material(self, mat):
        types = ["shaft", "flange", "plate"]
        vec = np.zeros(len(types), dtype=np.float32)
        if mat is None:
            return vec
        # mat could be Material or Product
        t = mat.type
        if t in types:
            vec[types.index(t)] = 1.0
        return vec

    def _get_obs_dict(self):
        # queues
        qx = float(self.station_X.queue_length())
        qy = float(self.station_Y.queue_length())
        qz = float(self.station_Z.queue_length())

        # mover_1 obs
        mover1_pos = self.mover1_pos.astype(np.float32)
        carry1_flag = 0.0 if self.mover1_carry is None else 1.0
        carry1_onehot = self._one_hot_material(self.mover1_carry)
        mover1_obs = np.concatenate([mover1_pos, [carry1_flag], carry1_onehot, [qx, qy, qz], [1.0 if self.machine_need_material else 0.0, 1.0 if self.machine_busy else 0.0]]).astype(np.float32)

        # mover_2 obs
        mover2_pos = self.mover2_pos.astype(np.float32)
        carry2_flag = 0.0 if self.mover2_carry is None else 1.0
        carry2_onehot = self._one_hot_material(self.mover2_carry)
        mover2_obs = np.concatenate([mover2_pos, [carry2_flag], carry2_onehot, [qx, qy, qz], [1.0 if self.machine_need_material else 0.0, 1.0 if self.machine_busy else 0.0]]).astype(np.float32)

        # manuf obs
        busy = 1.0 if self.machine_busy else 0.0
        timer = float(self.machine_timer)
        queue_y_len = float(self.station_Y.queue_length())
        # next required type: if machine_need_material we do not have required_type specific; for now set zeros
        required_onehot = np.zeros(3, dtype=np.float32)
        manuf_obs = np.concatenate([[busy], [timer], [queue_y_len], required_onehot, [1.0 if self.machine_need_material else 0.0]]).astype(np.float32)

        return {"mover_1": mover1_obs, "mover_2": mover2_obs, "manuf_1": manuf_obs}

    # expose action/obs spaces for external trainer
    def get_action_space(self, agent_id):
        return self.action_spaces[agent_id]

    def get_observation_space(self, agent_id):
        return self.observation_spaces[agent_id]