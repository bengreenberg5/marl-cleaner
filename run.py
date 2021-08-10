import os
import ray
from ray.tune import register_env
from typarse import BaseParser
import wandb

from cleaner.cleaner_env import *


class ArgParser(BaseParser):
    config: str = "simple_3x3"
    name: str = "cleaner"
    training_iters: int = 5
    checkpoint_freq: int = 0

    _help = {
        "config": "Path to the config of the experiment",
        "name": "Name of subdirectory containing results for this experiment",
        "training_iters": "Number of training iterations",
        "checkpoint_freq": "How many training iterations between checkpoints; "
                           "a value of 0 (default) disables checkpointing.",
        # "checkpoint_path": "Which checkpoint to load, if any",
    }


def main():
    args = ArgParser()
    config = load_config(args.config)
    env_config = config["env_config"]
    ray_config = config["ray_config"]
    run_config = config["run_config"]

    wandb.init(
        project=run_config["wandb_project"],
        entity=os.environ["USERNAME"],
        config=config,
        monitor_gym=True,
        sync_tensorboard=True,
    )  # integrate with Weights & Biases

    ray.shutdown()
    ray.init()
    register_env("ZSC-Cleaner", lambda _: CleanerEnv(env_config))

    results_dir = f"{os.path.expanduser('~')}/ray_results/{args.name}/"
    trainer = trainer_from_config(config, results_dir=results_dir)

    verbose = run_config["verbose"]

    # Training loop
    for i in range(args.training_iters):
        if run_config["verbose"]:
            print(f"starting training iteration {i}")
        result = trainer.train()

        if args.checkpoint_freq != 0 and i % args.checkpoint_freq == 0:
            save_path = save_trainer(trainer, config, path=results_dir, verbose=verbose)

    save_path = save_trainer(trainer, config, path=results_dir, verbose=verbose)

    ray.shutdown()


if __name__ == "__main__":
    main()
