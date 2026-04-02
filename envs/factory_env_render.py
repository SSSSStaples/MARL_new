def render(env):
    try:
        import matplotlib

        backend = str(matplotlib.get_backend()).lower()
        if backend == "agg":
            try:
                matplotlib.use("TkAgg", force=True)
            except Exception:
                pass
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for render(). Install it and retry.") from exc

    if env._viz["fig"] is None:
        plt.ion()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_aspect("equal")
        ax.set_xlim(env.bounds["xmin"], env.bounds["xmax"])
        ax.set_ylim(env.bounds["ymin"], env.bounds["ymax"])
        ax.set_title("Factory Env (multi-process)")
        env._viz["fig"] = fig
        env._viz["ax"] = ax
        try:
            fig.show()
        except Exception:
            pass

    fig = env._viz["fig"]
    ax = env._viz["ax"]
    ax.clear()
    ax.set_aspect("equal")
    ax.set_xlim(env.bounds["xmin"], env.bounds["xmax"])
    ax.set_ylim(env.bounds["ymin"], env.bounds["ymax"])
    ax.set_title(f"Factory Env (t={env.step_count})")

    for name, station in env.station_objs.items():
        x, y = station.position
        ax.scatter([x], [y], s=120, marker="s", color="#444444")
        ax.text(x + 0.2, y + 0.2, f"{name}\nq={station.queue_length()}", fontsize=8)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, agv_id in enumerate(env.agv_ids):
        x, y = env.agv_positions[i]
        color = colors[i % len(colors)]
        ax.scatter([x], [y], s=100, marker="o", color=color)
        label = agv_id
        carry = env.agv_carry[i]
        if carry:
            label += f"({carry.job_type})"
        ax.text(x + 0.2, y - 0.2, label, fontsize=8, color=color)

    y0 = env.bounds["ymax"] - 0.5
    for idx, process in enumerate(env.processes):
        machine = env.machines[process]
        status = "busy" if machine["busy"] else "idle"
        ax.text(env.bounds["xmin"] + 0.2, y0 - idx * 0.5, f"{process}: {status}", fontsize=8)

    fig.canvas.draw_idle()
    try:
        fig.canvas.flush_events()
    except Exception:
        pass
    try:
        plt.pause(0.001)
    except Exception:
        pass
    return fig


def close(env):
    if env._viz.get("fig") is not None:
        try:
            import matplotlib.pyplot as plt

            plt.close(env._viz["fig"])
        except Exception:
            pass
    env._viz = {"fig": None, "ax": None, "last_step": -1}
