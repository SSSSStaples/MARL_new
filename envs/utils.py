# env/utils.py
import numpy as np

def distance(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def move_towards(pos, target, speed):
    pos = np.array(pos)
    target = np.array(target)
    vec = target - pos
    dist = np.linalg.norm(vec)
    if dist < 1e-6:
        return pos
    step = min(speed, dist)
    return pos + vec / dist * step