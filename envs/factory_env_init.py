import numpy as np

from .station_objects import Order


def load_job_types(job_cfg):
    if job_cfg:
        return job_cfg
    return {
        "P1": {"part": "Shaft", "material": "Bar", "route": ["Turning", "Grinding"]},
        "P2": {"part": "Flange", "material": "Cube", "route": ["Turning", "Milling", "Drilling"]},
        "P3": {"part": "Plate", "material": "Plate", "route": ["Milling"]},
        "P4": {"part": "Frame", "material": "Bar", "route": ["Milling", "Drilling"]},
    }


def load_orders(orders_cfg):
    if orders_cfg:
        return orders_cfg
    return [
        {"id": 1, "job_type": "P1", "arrival": 15, "due": 160, "process_times": [35, 12]},
        {"id": 2, "job_type": "P2", "arrival": 50, "due": 180, "process_times": [32, 10, 20]},
        {"id": 3, "job_type": "P3", "arrival": 75, "due": 210, "process_times": [43]},
        {"id": 4, "job_type": "P4", "arrival": 100, "due": 230, "process_times": [38, 10]},
        {"id": 5, "job_type": "P1", "arrival": 130, "due": 260, "process_times": [47, 15]},
    ]


def init_internal_state(env):
    env.delivered_order_ids = set()
    env.agv_positions = [np.array(env.station_positions["SRC"], dtype=np.float32) for _ in env.agv_ids]
    env.agv_velocities = [np.zeros(2, dtype=np.float32) for _ in env.agv_ids]
    env.agv_carry = [None for _ in env.agv_ids]

    for s in env.station_objs.values():
        s.clear()

    for p in env.processes:
        env.machines[p]["busy"] = False
        env.machines[p]["timer"] = 0
        env.machines[p]["timer_init"] = 0
        env.machines[p]["order"] = None

    env.pending_orders = []
    for o in env.orders_template:
        job = env.job_types[o["job_type"]]
        env.pending_orders.append(
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
    env.pending_orders.sort(key=lambda x: x.arrival_time)

    if env.arrival_shift_to_first and env.pending_orders:
        shift = int(env.pending_orders[0].arrival_time) - 1
        if shift > 0:
            for o in env.pending_orders:
                o.arrival_time = int(o.arrival_time) - shift
                o.due_time = int(o.due_time) - shift

    env.completed_orders = 0
    env.step_count = 0
