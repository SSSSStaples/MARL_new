import numpy as np

from .station_objects import Order


def agv_target_position(env, agv_index, carry_before):
    if carry_before is not None:
        if getattr(carry_before, "finished", False):
            target_station = "SINK"
        else:
            next_proc = carry_before.next_process()
            target_station = "SINK" if next_proc is None else env.process_to_station.get(next_proc, "SINK")
        return np.array(env.station_positions[target_station], dtype=np.float32)

    pos = env.agv_positions[agv_index]
    # Heuristic target used only for:
    # - distance shaping (events["distance_shaping"])
    # - optional target-vector observation feature
    # 
    # Priority: if there are any finished orders waiting in non-SINK station queues,
    # navigate to the nearest such station to encourage timely delivery.
    finished_candidates = []
    candidates = []
    for s in env.stations:
        if s == "SINK":
            continue
        station_obj = env.station_objs[s]
        station_process = env.station_to_process.get(s, None)
        has_pickable = False
        has_finished = False
        for item in station_obj.queue:
            if not isinstance(item, Order):
                continue
            if getattr(item, "finished", False):
                has_finished = True
            if station_process is not None and (not item.finished) and item.next_process() == station_process:
                continue
            has_pickable = True
            break
        if has_finished:
            finished_candidates.append(s)
        if has_pickable:
            candidates.append(s)

    if finished_candidates:
        candidates = finished_candidates

    if not candidates:
        return np.array(env.station_positions["SRC"], dtype=np.float32)

    best = None
    best_dist = None
    for s in candidates:
        sp = np.array(env.station_positions[s], dtype=np.float32)
        d = float(np.linalg.norm(pos - sp))
        if best is None or d < best_dist:
            best = s
            best_dist = d
    return np.array(env.station_positions[best], dtype=np.float32)


def action_to_delta(action):
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


def move_by(env, pos, direction):
    direction = np.array(direction, dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return False
    proposed = pos + (direction / norm) * float(env.agv_speed)

    clamped = proposed.copy()
    clamped[0] = min(max(clamped[0], env.bounds["xmin"]), env.bounds["xmax"])
    clamped[1] = min(max(clamped[1], env.bounds["ymin"]), env.bounds["ymax"])

    out_of_bounds = not np.allclose(proposed, clamped)
    pos[:] = clamped
    return out_of_bounds


def boundary_shaping(env, pos, margin):
    margin = float(margin)
    if margin <= 0.0:
        return 0.0
    xmin, ymin, xmax, ymax = env.bounds["xmin"], env.bounds["ymin"], env.bounds["xmax"], env.bounds["ymax"]
    d_left = float(pos[0] - xmin)
    d_right = float(xmax - pos[0])
    d_bottom = float(pos[1] - ymin)
    d_top = float(ymax - pos[1])
    min_dist = min(d_left, d_right, d_bottom, d_top)
    if min_dist >= margin:
        return 0.0
    risk = (margin - min_dist) / margin
    return -float(risk)


def near(pos, target, radius):
    return np.linalg.norm(pos - target) <= radius


def current_station(env, pos, radius):
    for name, station in env.station_objs.items():
        if near(pos, np.array(station.position, dtype=np.float32), radius=radius):
            return name
    return None


def is_valid_drop(env, station, order):
    if order.finished:
        return station == "SINK"
    next_proc = order.next_process()
    if next_proc is None:
        return station == "SINK"
    return station == env.process_to_station.get(next_proc)


def can_accept_drop(env, station, order):
    """
    Enforce per-process eligible waiting buffer capacity.

    We only count *eligible waiting* orders for the station's process:
      - order.next_process() == station_process
      - and not finished
    """
    cap = int(getattr(env, "eligible_buffer_capacity", 0) or 0)
    if cap <= 0:
        return True
    if station == "SINK":
        return True
    station_process = env.station_to_process.get(station, None)
    if station_process is None:
        return True
    if order is None or getattr(order, "finished", False):
        return True
    if order.next_process() != station_process:
        return True

    station_obj = env.station_objs[station]
    eligible_waiting = 0
    for item in station_obj.queue:
        if isinstance(item, Order) and (not item.finished) and item.next_process() == station_process:
            eligible_waiting += 1
    return eligible_waiting < cap


def pop_order_for_pick(env, station, pick_rank, requester_id=None):
    station_process = env.station_to_process.get(station.name, None)

    candidates = []
    for idx, item in enumerate(station.queue):
        if not isinstance(item, Order):
            continue
        # Respect handoff reservation: do not allow picking a finished order reserved by another AGV.
        if requester_id is not None and getattr(item, "finished", False):
            rb = getattr(item, "reserved_by", None)
            if rb is not None and str(rb) != str(requester_id):
                continue
        # Do not allow picking items that are currently eligible waiting input for this station's machine.
        if station_process is not None and (not item.finished) and item.next_process() == station_process:
            continue
        candidates.append((idx, item))

    # Rank rule: prioritize unclaimed finished handoff requests first, then everything else
    # (preserve original queue scan order within each group).
    finished_unclaimed = [(idx, it) for idx, it in candidates if getattr(it, "finished", False) and getattr(it, "reserved_by", None) is None]
    others = [(idx, it) for idx, it in candidates if (idx, it) not in finished_unclaimed]
    candidates = finished_unclaimed + others
    if not candidates:
        return None

    # Scheme B: slot-based selection.
    # The station.queue order defines the buffer order (no heuristic sorting).
    ranked = candidates

    if pick_rank < 0 or pick_rank >= len(ranked):
        return None
    remove_idx, _ = ranked[pick_rank]
    return station.queue.pop(remove_idx)


def apply_agv_actions(env, action_dict, events):
    boundary_vals = {}
    distance_shaping = {}

    for i, agv_id in enumerate(env.agv_ids):
        action = int(action_dict.get(agv_id, 0))
        pos = env.agv_positions[i]
        prev = pos.copy()
        carry_before = env.agv_carry[i]
        env.agv_velocities[i][:] = 0.0

        if 1 <= action <= 8:
            delta = action_to_delta(action)
            if move_by(env, pos, delta):
                events["agv_out_of_bounds"].append(agv_id)
            env.agv_velocities[i][:] = (pos - prev).astype(np.float32)

            target_pos = agv_target_position(env, i, carry_before)
            if target_pos is not None:
                dist_prev = float(np.linalg.norm(prev - target_pos))
                dist_new = float(np.linalg.norm(pos - target_pos))
                delta_dist = dist_prev - dist_new
                if env.distance_shaping_mode in {"progress", "pos", "positive"}:
                    delta_dist = max(0.0, delta_dist)
                distance_shaping[agv_id] = distance_shaping.get(agv_id, 0.0) + float(delta_dist)

        elif env._pick_action_start <= action <= env._pick_action_end:
            if env.agv_carry[i] is None:
                station = current_station(env, pos, radius=env.pickup_radius)
                if station and station != "SINK":
                    pick_rank = int(action - env._pick_action_start)
                    item = pop_order_for_pick(env, env.station_objs[station], pick_rank, requester_id=agv_id)
                    if item is not None:
                        env.agv_carry[i] = item
                        events["agv_picked"].append(agv_id)
                        if getattr(item, "finished", False):
                            # Handoff claim: first AGV to pick a finished order becomes its owner.
                            if getattr(item, "handoff_requested", False) and getattr(item, "reserved_by", None) is None:
                                item.reserved_by = str(agv_id)
                                item.reserved_until = None
                                events.setdefault("handoff_claimed", []).append(
                                    {"agent": str(agv_id), "order_id": int(getattr(item, "order_id", -1))}
                                )
                            events.setdefault("agv_picked_finished", []).append(agv_id)
                    else:
                        events.setdefault("agv_pick_failed", []).append(agv_id)
                else:
                    events.setdefault("agv_pick_failed", []).append(agv_id)
            else:
                events.setdefault("agv_pick_failed", []).append(agv_id)

        elif action == env._drop_action:
            if env.agv_carry[i] is None:
                events.setdefault("agv_drop_failed", []).append(agv_id)
            else:
                station = current_station(env, pos, radius=env.drop_radius)
                if not station:
                    events.setdefault("agv_drop_failed", []).append(agv_id)
                elif not is_valid_drop(env, station, env.agv_carry[i]):
                    events["agv_wrong_drop"].append(agv_id)
                else:
                    if station == "SINK":
                        late = max(0, env.step_count - env.agv_carry[i].due_time)
                        delivered_order = env.agv_carry[i]
                        events["agv_delivered"].append(
                            {
                                "agent": agv_id,
                                "late_steps": late,
                                "delivered_step": int(env.step_count),
                                "due_time": int(delivered_order.due_time),
                                "order_id": int(delivered_order.order_id),
                                "job_type": str(delivered_order.job_type),
                            }
                        )
                        env.completed_orders += 1
                        env.agv_carry[i] = None
                    else:
                        if can_accept_drop(env, station, env.agv_carry[i]):
                            env.station_objs[station].add_item(env.agv_carry[i])
                            env.agv_carry[i] = None
                            events["agv_correct_drop"].append(agv_id)
                        else:
                            events.setdefault("agv_buffer_full", []).append(agv_id)

        if env.boundary_margin > 0.0:
            boundary_vals[agv_id] = float(boundary_shaping(env, env.agv_positions[i], env.boundary_margin))

    if boundary_vals:
        events["boundary_shaping"] = boundary_vals
    if distance_shaping:
        events["distance_shaping"] = distance_shaping

    if env.collision_radius > 0.0 and len(env.agv_ids) >= 2:
        collision_agents = set()
        r2 = float(env.collision_radius) * float(env.collision_radius)
        for i in range(len(env.agv_ids)):
            for j in range(i + 1, len(env.agv_ids)):
                d = env.agv_positions[i] - env.agv_positions[j]
                if float(d[0] * d[0] + d[1] * d[1]) <= r2:
                    collision_agents.add(env.agv_ids[i])
                    collision_agents.add(env.agv_ids[j])
        if collision_agents:
            events["collision_agents"] = list(sorted(collision_agents))
            events["collision_count"] = int(len(collision_agents))

    if env.global_task_complete_reward:
        completions = int(len(events.get("agv_delivered", [])))
        if completions > 0:
            events["global_task_complete"] = completions
