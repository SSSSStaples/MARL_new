# config/config.py

WORLD_SIZE = (100, 100)

SOURCE_POS = (0, 0)
FACTORY_POS = {
    0: (50, 30),
    1: (50, 60),
    2: (50, 90),
}
SINK_POS = (100, 0)

PROCESS_TIME = {
    0: 3,
    1: 4,
    2: 2
}

STEP_SPEED = 5.0
PICK_DIST = 2.0
MAX_STEPS = 500

INIT_MATERIALS = 12