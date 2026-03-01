import random
from agents.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    def act(self, obs):
        return random.randint(0,6)