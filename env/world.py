# env/world.py
from config.config import PROCESS_TIME, INIT_MATERIALS

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


class World:
    """Container for the simulation state used by :class:`FactoryEnv`.

    Attributes:
        time (int): current time step.
        factories (dict[int, Factory]): each factory instance keyed by type.
        material_source (dict[int, int]): remaining raw materials at the source.
        sink_count (int): number of materials delivered to the sink.
    """

    def __init__(self):
        self.time = 0
        # create one factory for each material type defined in PROCESS_TIME
        self.factories = {i: Factory(i) for i in PROCESS_TIME}
        # initial amount of raw material of each type
        self.material_source = {i: INIT_MATERIALS for i in PROCESS_TIME}
        self.sink_count = 0

    def reset(self):
        """Reset the world to an initial state.

        Called from ``FactoryEnv.reset`` before each episode.
        """
        self.time = 0
        for f in self.factories.values():
            f.queue.clear()
            f.processing = None
            f.timer = 0
            f.ready = 0
        self.material_source = {i: INIT_MATERIALS for i in PROCESS_TIME}
        self.sink_count = 0
