import argparse
import time

from envs.factory_env import FactoryEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--sleep_s", type=float, default=0.05, help="Extra sleep per rendered frame (in seconds).")
    parser.add_argument("--render_every", type=int, default=1, help="Render every N env steps (1 = every step).")
    args = parser.parse_args()

    env = FactoryEnv(config_path=args.config)
    obs = env.reset()

    episode = 0
    steps_in_episode = 0

    # Keep running until user interrupts (Ctrl+C).
    while True:
        action_dict = {agent_id: env.get_action_space(agent_id).sample() for agent_id in env.agent_ids}
        obs, rewards, dones, infos = env.step(action_dict)
        steps_in_episode += 1

        if args.render_every > 0 and (steps_in_episode % int(args.render_every) == 0):
            env.render()
            if args.sleep_s > 0:
                time.sleep(float(args.sleep_s))

        if dones.get("__all__", False):
            print(
                f"[Episode {episode}] steps={steps_in_episode} completed_orders={infos.get('completed_orders')} time={infos.get('time')}"
            )
            episode += 1
            steps_in_episode = 0
            obs = env.reset()


if __name__ == "__main__":
    main()
