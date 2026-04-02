import numpy as np

from .station_objects import Order


def one_hot(items, value):
    vec = np.zeros(len(items), dtype=np.float32)
    if value in items:
        vec[items.index(value)] = 1.0
    return vec


def get_obs_dict(env):
    obs = {}
    queue_lens = [float(env.station_objs[s].queue_length()) for s in env.stations]
    machine_busy = [1.0 if env.machines[p]["busy"] else 0.0 for p in env.processes]
    machine_timer = None
    if env.obs_include_machine_timer:
        machine_timer = [
            (float(env.machines[p]["timer"]) / float(max(1, int(env.machines[p].get("timer_init", 0))))) if env.machines[p]["busy"] else 0.0
            for p in env.processes
        ]
    time_frac = float(env.step_count) / float(max(1, env.max_steps))

    # Handoff alert board (AGV-visible): which stations have finished orders waiting, and waiting age.
    handoff_exists = None
    handoff_age = None
    if getattr(env, "obs_include_handoff_alert", False):
        stations = [s for s in env.stations if s != "SINK"]
        exists = np.zeros(len(stations), dtype=np.float32)
        age = np.zeros(len(stations), dtype=np.float32)
        denom = float(max(1, int(env.max_steps)))
        for idx, s in enumerate(stations):
            st = env.station_objs[s]
            best_age = 0.0
            has_finished = False
            for item in st.queue:
                if not isinstance(item, Order):
                    continue
                if not getattr(item, "finished", False):
                    continue
                # Only alert on "unclaimed" finished orders (handoff requested but not yet owned).
                if getattr(item, "reserved_by", None) is not None:
                    continue
                has_finished = True
                fs = getattr(item, "finished_step", None)
                if fs is None:
                    a = 0.0
                else:
                    a = float(max(0, int(env.step_count) - int(fs)))
                best_age = max(best_age, a)
            exists[idx] = 1.0 if has_finished else 0.0
            age[idx] = float(best_age) / denom
        handoff_exists = exists
        if getattr(env, "obs_handoff_include_age", True):
            handoff_age = age

    job_keys = list(env.job_types.keys())
    for i, agv_id in enumerate(env.agv_ids):
        pos = env.agv_positions[i].astype(np.float32)
        vel = env.agv_velocities[i].astype(np.float32)

        station_pos_feat = None
        if env.obs_include_station_positions:
            xmin, ymin, xmax, ymax = env.bounds["xmin"], env.bounds["ymin"], env.bounds["xmax"], env.bounds["ymax"]
            span_x = float(max(1e-6, xmax - xmin))
            span_y = float(max(1e-6, ymax - ymin))
            feats = []
            for s in env.stations:
                sx, sy = env.station_positions[s]
                if env.obs_station_pos_mode in {"absolute", "abs"}:
                    x, y = float(sx), float(sy)
                    if env.obs_station_pos_normalize:
                        x = (x - float(xmin)) / span_x
                        y = (y - float(ymin)) / span_y
                    feats.extend([x, y])
                else:
                    dx = float(sx) - float(pos[0])
                    dy = float(sy) - float(pos[1])
                    if env.obs_station_pos_normalize:
                        dx = dx / span_x
                        dy = dy / span_y
                    feats.extend([dx, dy])
            station_pos_feat = np.array(feats, dtype=np.float32)

        neighbor_feat = None
        if env.obs_include_neighbor_agvs and int(env.obs_neighbor_k) > 0:
            xmin, ymin, xmax, ymax = env.bounds["xmin"], env.bounds["ymin"], env.bounds["xmax"], env.bounds["ymax"]
            span_x = float(max(1e-6, xmax - xmin))
            span_y = float(max(1e-6, ymax - ymin))

            others = []
            for j in range(len(env.agv_ids)):
                if j == i:
                    continue
                other_pos = env.agv_positions[j]
                dx = float(other_pos[0] - pos[0])
                dy = float(other_pos[1] - pos[1])
                dist2 = dx * dx + dy * dy
                others.append((dist2, dx, dy))
            others.sort(key=lambda x: x[0])

            k = int(env.obs_neighbor_k)
            feats = np.zeros(2 * k, dtype=np.float32)
            for idx in range(min(k, len(others))):
                _, dx, dy = others[idx]
                if env.obs_neighbor_normalize:
                    dx = dx / span_x
                    dy = dy / span_y
                feats[2 * idx] = float(dx)
                feats[2 * idx + 1] = float(dy)
            neighbor_feat = feats

        target_feat = None
        if env.obs_include_target_vector:
            xmin, ymin, xmax, ymax = env.bounds["xmin"], env.bounds["ymin"], env.bounds["xmax"], env.bounds["ymax"]
            span_x = float(max(1e-6, xmax - xmin))
            span_y = float(max(1e-6, ymax - ymin))
            target_pos = env._agv_target_position(i, env.agv_carry[i])
            if target_pos is None:
                target_feat = np.zeros(2, dtype=np.float32)
            else:
                dx = float(target_pos[0] - pos[0])
                dy = float(target_pos[1] - pos[1])
                if env.obs_target_normalize:
                    dx = dx / span_x
                    dy = dy / span_y
                target_feat = np.array([dx, dy], dtype=np.float32)

        carry = env.agv_carry[i]
        carry_flag = 0.0 if carry is None else 1.0
        carry_job = one_hot(job_keys, carry.job_type) if carry else np.zeros(len(job_keys), dtype=np.float32)
        next_proc = carry.next_process() if carry else None
        carry_next = one_hot(env.processes, next_proc) if next_proc else np.zeros(len(env.processes), dtype=np.float32)

        current_station = env._current_station(pos, radius=env.obs_station_radius)
        cur_station_oh = (
            one_hot(env.stations, current_station)
            if (env.obs_include_current_station and current_station is not None)
            else (np.zeros(len(env.stations), dtype=np.float32) if env.obs_include_current_station else None)
        )

        pick_window = None
        if env.obs_pick_window_k > 0:
            pick_window = np.zeros(int(env.obs_pick_window_k) * (1 + 1 + len(env.processes)), dtype=np.float32)
            if current_station is not None and current_station in env.station_objs:
                station_obj = env.station_objs[current_station]
                station_process = env.station_to_process.get(current_station, None)
                candidates = []
                for item in station_obj.queue:
                    if not isinstance(item, Order):
                        continue
                    if station_process is not None and (not item.finished) and item.next_process() == station_process:
                        continue
                    candidates.append(item)

                if candidates:
                    # Scheme B: slot-based selection.
                    # The station queue scan order defines the buffer order (no heuristic sorting).
                    ranked = candidates

                    for j in range(min(int(env.obs_pick_window_k), len(ranked))):
                        o = ranked[j]
                        due_remaining = float(o.due_time - env.step_count) / float(max(1, env.max_steps))
                        priority = float(getattr(o, "priority", 0))
                        np_next = o.next_process()
                        next_oh = one_hot(env.processes, np_next) if np_next else np.zeros(len(env.processes), dtype=np.float32)
                        start = j * (1 + 1 + len(env.processes))
                        pick_window[start] = due_remaining
                        pick_window[start + 1] = priority
                        pick_window[start + 2 : start + 2 + len(env.processes)] = next_oh

        obs_vec = np.concatenate(
            [
                pos,
                vel,
                *([station_pos_feat] if station_pos_feat is not None else []),
                *([neighbor_feat] if neighbor_feat is not None else []),
                *([target_feat] if target_feat is not None else []),
                [carry_flag],
                carry_job,
                carry_next,
                np.array(queue_lens, dtype=np.float32),
                np.array(machine_busy, dtype=np.float32),
                *([np.array(machine_timer, dtype=np.float32)] if machine_timer is not None else []),
                [time_frac],
                *([cur_station_oh] if cur_station_oh is not None else []),
                *([handoff_exists] if handoff_exists is not None else []),
                *([handoff_age] if handoff_age is not None else []),
                *([pick_window] if pick_window is not None else []),
            ]
        ).astype(np.float32)
        obs[agv_id] = obs_vec

    if env.machine_as_agent:
        proc_denom = float(max(1, int(getattr(env, "process_time_max", 1))))
        for process in env.processes:
            mid = env.process_to_machine_id[process]
            machine = env.machines[process]
            busy = 1.0 if machine["busy"] else 0.0
            timer_frac = float(machine["timer"]) / float(max(1, int(machine.get("timer_init", 0)))) if machine["busy"] else 0.0

            station = env.station_objs[env.process_to_station[process]]
            eligible = [item for item in station.queue if isinstance(item, Order) and item.next_process() == process]
            eligible_q_len = float(len(eligible))

            topk = np.zeros(int(env.machine_obs_k) * 3, dtype=np.float32)
            if eligible:
                # Scheme B: slot-based selection.
                # The eligible scan order defines the buffer order (no heuristic sorting).
                ranked = eligible

                default_t = int(env.process_time_defaults.get(process, 10))
                for j in range(min(int(env.machine_obs_k), len(ranked))):
                    o = ranked[j]
                    due_remaining = float(o.due_time - env.step_count) / float(max(1, env.max_steps))
                    priority = float(getattr(o, "priority", 0))
                    pt = float(o.current_process_time(default_t)) / proc_denom
                    start = j * 3
                    topk[start] = due_remaining
                    topk[start + 1] = priority
                    topk[start + 2] = pt

            obs[mid] = np.concatenate([[busy], [timer_frac], [eligible_q_len], [time_frac], topk]).astype(np.float32)

    return obs
