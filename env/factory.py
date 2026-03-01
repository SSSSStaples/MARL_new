# env/factory.py
from config.config import PROCESS_TIME

class Factory:

    def __init__(self, factory_type):
        self.type = factory_type
        self.queue = []
        self.processing = None
        self.timer = 0
        self.ready = 0

    def step(self):
        if self.processing is None:
            if self.queue:
                self.processing = self.queue.pop(0)
                self.timer = PROCESS_TIME[self.processing]
        else:
            self.timer -= 1
            if self.timer <= 0:
                self.ready += 1
                self.processing = None
                self.timer = 0