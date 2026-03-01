# env/factory_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config.config import *
from env.world import World
from env.utils import move_towards, distance


class FactoryEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.world = World()

        self.agents = ["mover1", "mover2"]

        # 每个 mover 19 维
        obs_dim = 19

        self.observation_space = {
            "mover1": spaces.Box(-1e6, 1e6, shape=(obs_dim,), dtype=np.float32),
            "mover2": spaces.Box(-1e6, 1e6, shape=(obs_dim,), dtype=np.float32)
        }

        self.action_space = {
            "mover1": spaces.Discrete(7),
            "mover2": spaces.Discrete(7)
        }

        self.reset()

    def reset(self, seed=None, options=None):
        self.world.reset()

        self.pos = {
            "mover1": np.array(SOURCE_POS, dtype=float),
            "mover2": np.array((60, 60), dtype=float)
        }

        self.carry = {
            "mover1": -1,
            "mover2": -1
        }

        self.target = {
            "mover1": None,
            "mover2": None
        }

        return self._get_obs(), {}

    def step(self, actions):

        self.world.time += 1
        rewards = {a: -1.0 for a in self.agents}  # makespan优化
        done = False

        self._handle_actions(actions)
        self._move_agents()
        self._handle_pick_drop(actions)

        for f in self.world.factories.values():
            f.step()

        if self.world.sink_count >= INIT_MATERIALS:
            done = True
            rewards = {a: r + 100 for a, r in rewards.items()}

        if self.world.time >= MAX_STEPS:
            done = True

        return self._get_obs(), rewards, done, False, {}

    # ---------------- ACTIONS ----------------

    def _handle_actions(self, actions):

        a1 = actions["mover1"]
        if a1 == 1:
            self.target["mover1"] = SOURCE_POS
        elif a1 in [2,3,4]:
            self.target["mover1"] = FACTORY_POS[a1-2]

        a2 = actions["mover2"]
        if a2 in [1,2,3]:
            self.target["mover2"] = FACTORY_POS[a2-1]
        elif a2 == 4:
            self.target["mover2"] = SINK_POS

    def _move_agents(self):
        for ag in self.agents:
            if self.target[ag] is not None:
                self.pos[ag] = move_towards(self.pos[ag], self.target[ag], STEP_SPEED)

    def _handle_pick_drop(self, actions):

        # mover1 pick
        if actions["mover1"] == 5:
            if distance(self.pos["mover1"], SOURCE_POS) < PICK_DIST:
                for t in self.world.material_source:
                    if self.world.material_source[t] > 0:
                        self.world.material_source[t] -= 1
                        self.carry["mover1"] = t
                        break

        # mover1 drop
        if actions["mover1"] == 6:
            t = self.carry["mover1"]
            if t != -1 and distance(self.pos["mover1"], FACTORY_POS[t]) < PICK_DIST:
                self.world.factories[t].queue.append(t)
                self.carry["mover1"] = -1

        # mover2 pick
        if actions["mover2"] == 5:
            for t in self.world.factories:
                if distance(self.pos["mover2"], FACTORY_POS[t]) < PICK_DIST:
                    if self.world.factories[t].ready > 0:
                        self.world.factories[t].ready -= 1
                        self.carry["mover2"] = t
                        break

        # mover2 drop
        if actions["mover2"] == 6:
            if self.carry["mover2"] != -1 and distance(self.pos["mover2"], SINK_POS) < PICK_DIST:
                self.world.sink_count += 1
                self.carry["mover2"] = -1

    # ---------------- OBSERVATION ----------------

    def _get_obs(self):

        obs = {}

        S_counts = np.array([
            self.world.material_source[0],
            self.world.material_source[1],
            self.world.material_source[2]
        ], dtype=np.float32)

        factory_features = []

        for t in [0,1,2]:
            f = self.world.factories[t]
            processing_flag = 1.0 if f.processing is not None else 0.0
            factory_features.extend([
                float(f.ready),
                float(len(f.queue)),
                processing_flag,
                float(f.timer)
            ])

        factory_features = np.array(factory_features, dtype=np.float32)

        for ag in self.agents:

            pos = self.pos[ag]
            carry = float(self.carry[ag])

            obs_vec = np.concatenate([
                pos.astype(np.float32),          # x,y
                np.array([carry], dtype=np.float32),
                S_counts,
                factory_features,
                np.array([float(self.world.time)], dtype=np.float32)
            ])

            obs[ag] = obs_vec

        return obs