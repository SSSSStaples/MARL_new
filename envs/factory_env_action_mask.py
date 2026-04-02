import numpy as np

from .factory_env_agv import action_to_delta, current_station, is_valid_drop, can_accept_drop
from .station_objects import Order


def get_action_mask(env, agent_id):
    if agent_id not in env.action_spaces:
        return None
    n = int(env.action_spaces[agent_id].n)
    mask = np.ones((n,), dtype=np.float32)

    if agent_id in env.agv_ids:
        i = env.agv_ids.index(agent_id)
        pos = env.agv_positions[i]
        carry = env.agv_carry[i]

        for a in range(1, 9):
            delta = action_to_delta(a)
            d = np.array(delta, dtype=np.float32)
            norm = float(np.linalg.norm(d))
            if norm < 1e-6:
                mask[a] = 0.0
                continue
            proposed = pos + (d / norm) * float(env.agv_speed)
            clamped = proposed.copy()
            clamped[0] = min(max(clamped[0], env.bounds["xmin"]), env.bounds["xmax"])
            clamped[1] = min(max(clamped[1], env.bounds["ymin"]), env.bounds["ymax"])
            if not np.allclose(proposed, clamped):
                mask[a] = 0.0

        if carry is not None:
            mask[env._pick_action_start : env._pick_action_end + 1] = 0.0
        else:
            station = current_station(env, pos, radius=env.pickup_radius)
            if (station is None) or (station == "SINK"):
                mask[env._pick_action_start : env._pick_action_end + 1] = 0.0
            else:
                # Build ranked pick candidates using the same logic as the actual picker.
                station_obj = env.station_objs[station]
                # Reconstruct candidates list by mimicking pop_order_for_pick ranking.
                station_process = env.station_to_process.get(station_obj.name, None)
                raw = []
                for idx, item in enumerate(station_obj.queue):
                    if not isinstance(item, Order):
                        continue
                    if getattr(item, "finished", False):
                        rb = getattr(item, "reserved_by", None)
                        if rb is not None and str(rb) != str(agent_id):
                            continue
                    if station_process is not None and (not item.finished) and item.next_process() == station_process:
                        continue
                    raw.append(item)
                finished_unclaimed = [o for o in raw if getattr(o, "finished", False) and getattr(o, "reserved_by", None) is None]
                others = [o for o in raw if o not in finished_unclaimed]
                candidates = finished_unclaimed + others

                if len(candidates) == 0:
                    mask[env._pick_action_start : env._pick_action_end + 1] = 0.0
                else:
                    # Optional training aid: if there is an unclaimed finished handoff at this station,
                    # force picking it (and only it) when the AGV is empty and in pickup range.
                    if getattr(env, "force_pick_finished_handoff", False) and len(finished_unclaimed) > 0:
                        # Force pick ranks that correspond to finished_unclaimed (they are first).
                        max_rank = min(len(finished_unclaimed) - 1, int(env._pick_action_end - env._pick_action_start))
                        mask[:] = 0.0
                        for r in range(0, max_rank + 1):
                            a = int(env._pick_action_start + r)
                            if 0 <= a < n:
                                mask[a] = 1.0
                    elif getattr(env, "force_pick_at_station", False):
                        # Force picking *something* (rank 0) to keep the pipeline moving.
                        mask[:] = 0.0
                        a = int(env._pick_action_start)
                        if 0 <= a < n:
                            mask[a] = 1.0
                    else:
                        for a in range(env._pick_action_start, env._pick_action_end + 1):
                            rank = int(a - env._pick_action_start)
                            if rank >= len(candidates):
                                mask[a] = 0.0

        if carry is None:
            mask[env._drop_action] = 0.0
        else:
            station = current_station(env, pos, radius=env.drop_radius)
            if station is None:
                mask[env._drop_action] = 0.0
            else:
                if not is_valid_drop(env, station, carry):
                    mask[env._drop_action] = 0.0
                # If the target station buffer is full, dropping would fail; mask it out.
                if float(mask[env._drop_action]) > 0.0 and station != "SINK":
                    if not can_accept_drop(env, station, carry):
                        mask[env._drop_action] = 0.0
                # Optional training aid: if at SINK with a finished order, force DROP as the only action.
                if (
                    getattr(env, "force_drop_at_sink", False)
                    and station == "SINK"
                    and getattr(carry, "finished", False)
                    and float(mask[env._drop_action]) > 0.0
                ):
                    mask[:] = 0.0
                    mask[env._drop_action] = 1.0
                # Optional training aid: if at a valid drop station with any order, force DROP.
                if (
                    getattr(env, "force_drop_when_valid", False)
                    and float(mask[env._drop_action]) > 0.0
                ):
                    mask[:] = 0.0
                    mask[env._drop_action] = 1.0

        return mask

    if env.machine_as_agent and agent_id in env.machine_ids:
        process = str(agent_id).replace("machine_", "", 1)
        machine = env.machines.get(process)
        if machine is None:
            return mask
        k = int(getattr(env, "machine_action_k", 0) or 0)
        idle_action = int(k)  # action K means idle (see envs/factory_env_machine.py)
        if machine["busy"]:
            mask[:] = 0.0
            # When busy, any action is a no-op; prefer the explicit idle action for stability.
            if 0 <= idle_action < n:
                mask[idle_action] = 1.0
            else:
                mask[0] = 1.0
            return mask

        station = env.station_objs[env.process_to_station[process]]
        eligible = [item for item in station.queue if isinstance(item, Order) and item.next_process() == process]
        eligible_n = int(len(eligible))
        mask[:] = 0.0
        if eligible_n <= 0:
            # No eligible work: only allow idle.
            if 0 <= idle_action < n:
                mask[idle_action] = 1.0
            else:
                mask[0] = 1.0
        else:
            # Force machines to pick a valid eligible slot (prevents collapse to always-idle).
            # Valid pick actions are 0..min(K-1, eligible_n-1). Idle is disallowed when work exists.
            max_pick = min(int(k) - 1, eligible_n - 1)
            for a in range(0, max_pick + 1):
                if 0 <= a < n:
                    mask[a] = 1.0
        return mask

    return mask


def get_action_masks(env):
    masks = {}
    for a in env.agent_ids:
        m = get_action_mask(env, a)
        if m is not None:
            masks[a] = m
    return masks
