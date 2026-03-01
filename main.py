# main.py

import argparse
from trainer.ppo_trainer import PPOTrainer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "eval"])
    parser.add_argument("--timesteps", type=int, default=300000)
    parser.add_argument("--model_path", type=str, default="models/ppo_final.zip")

    args = parser.parse_args()

    trainer = PPOTrainer(
        total_timesteps=args.timesteps
    )

    if args.mode == "train":
        print("Starting PPO training...")
        trainer.train()
        print("Training completed.")
        print("Now evaluating trained model...")
        trainer.evaluate(episodes=3)

    elif args.mode == "eval":
        print("Evaluating existing model...")
        trainer.model = trainer.model.load(args.model_path)
        trainer.evaluate(episodes=5)


if __name__ == "__main__":
    main()