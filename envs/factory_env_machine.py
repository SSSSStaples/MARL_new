from .station_objects import Order


def pop_next_for_process(station, process):
    for idx, item in enumerate(station.queue):
        if isinstance(item, Order) and item.next_process() == process:
            return station.queue.pop(idx)
    return None


def pop_order_for_process_by_rank(env, station, process, pick_rank, return_meta=False):
    candidates = [(idx, item) for idx, item in enumerate(station.queue) if isinstance(item, Order) and item.next_process() == process]
    if not candidates:
        if return_meta:
            return None, {"action": 0, "pick_rank": int(pick_rank), "eligible_count": 0}
        return None

    # Scheme B: slot-based selection.
    # The eligible scan order defines the buffer order (no heuristic sorting).
    ranked = candidates

    pick_rank = int(pick_rank)
    if pick_rank < 0 or pick_rank >= len(ranked):
        if return_meta:
            return None, {"action": 0, "pick_rank": int(pick_rank), "eligible_count": int(len(ranked))}
        return None
    remove_idx, _ = ranked[pick_rank]
    order = station.queue.pop(remove_idx)
    if return_meta:
        return order, {"action": int(pick_rank), "pick_rank": int(pick_rank), "eligible_count": int(len(ranked))}
    return order


def machine_step(env, action_dict=None):
    action_dict = action_dict or {}
    selections = []
    machine_events = {
        "machine_started": [],
        "machine_completed": [],
        "machine_order_advanced": [],
        "machine_order_finished": [],
        "handoff_requested": [],
        "machine_select_failed": [],
    }

    for process in env.processes:
        machine = env.machines[process]
        if not machine["busy"]:
            station = env.station_objs[env.process_to_station[process]]
            if env.machine_as_agent:
                mid = env.process_to_machine_id[process]
                act = int(action_dict.get(mid, 0))
                # Scheme B:
                # - action 0..K-1 chooses a slot in the eligible waiting buffer (scan order)
                # - action K means idle (do nothing)
                k = int(getattr(env, "machine_action_k", 0) or 0)
                if act >= k:
                    order, meta = None, {"action": int(act), "pick_rank": -1, "eligible_count": 0}
                else:
                    order, meta = pop_order_for_process_by_rank(env, station, process, act, return_meta=True)
            else:
                order = pop_next_for_process(station, process)
                meta = None

            if order is not None:
                machine["busy"] = True
                machine["order"] = order
                default_t = env.process_time_defaults.get(process, 10)
                machine["timer"] = order.current_process_time(default_t)
                machine["timer_init"] = int(machine["timer"])
                if env.machine_as_agent:
                    selections.append(
                        {
                            "machine_id": env.process_to_machine_id[process],
                            "process": process,
                            "action": int(meta.get("action", 0) if meta else 0),
                            "pick_rank": int(meta.get("pick_rank", 0) if meta else 0),
                            "eligible_count": int(meta.get("eligible_count", 0) if meta else 0),
                            "chosen_order_id": int(order.order_id),
                        }
                    )
                machine_events["machine_started"].append(env.process_to_machine_id[process])
            elif env.machine_as_agent:
                # Penalize only *truly invalid* selections:
                # - if there exists at least one eligible order, but the chosen slot is out of range.
                # If there are 0 eligible orders, treat any selection as "no-op" (otherwise the optimal
                # behavior collapses to always choosing the explicit idle action without exploration).
                eligible_count = int(meta.get("eligible_count", 0) if isinstance(meta, dict) else 0)
                if eligible_count > 0 and int(meta.get("pick_rank", -1)) >= eligible_count:
                    machine_events["machine_select_failed"].append(env.process_to_machine_id[process])

    for process in env.processes:
        machine = env.machines[process]
        if machine["busy"]:
            machine["timer"] -= 1
            if machine["timer"] <= 0:
                order = machine["order"]
                order.advance()
                mid = env.process_to_machine_id[process]
                machine_events["machine_order_advanced"].append(
                    {"machine_id": mid, "order_id": int(order.order_id), "process": str(process)}
                )
                if getattr(order, "finished", False):
                    if getattr(order, "finished_step", None) is None:
                        try:
                            order.finished_step = int(getattr(env, "step_count", 0))
                        except Exception:
                            order.finished_step = 0
                    if not getattr(order, "handoff_requested", False):
                        order.handoff_requested = True
                        order.reserved_by = None
                        order.reserved_until = None
                    machine_events["machine_order_finished"].append(
                        {"machine_id": mid, "order_id": int(order.order_id), "process": str(process)}
                    )
                    try:
                        st_name = env.process_to_station.get(process, None)
                        if st_name is None:
                            st_name = str(env.process_to_station[process])
                    except Exception:
                        st_name = str(env.process_to_station.get(process, ""))
                    machine_events["handoff_requested"].append(
                        {
                            "order_id": int(order.order_id),
                            "station": str(st_name),
                            "due_time": int(getattr(order, "due_time", 0) or 0),
                            "requested_step": int(getattr(env, "step_count", 0) or 0),
                        }
                    )
                station = env.station_objs[env.process_to_station[process]]
                station.add_item(order)
                machine["busy"] = False
                machine["timer"] = 0
                machine["timer_init"] = 0
                machine["order"] = None
                machine_events["machine_completed"].append(env.process_to_machine_id[process])

    return selections, machine_events
