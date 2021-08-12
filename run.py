from collections import defaultdict
from typing import Dict, Optional

from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.typing import PolicyID
from typarse import BaseParser
import wandb

import ray
from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.agents import DefaultCallbacks
from ray.tune import register_env

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
    }


def evaluate(config, checkpoint_path, record=True):
    results_dir = "tmp"  # TODO
    trainer = trainer_from_config(config, results_dir)
    done = {"__all__": False}
    env = CleanerEnv(config["env_config"])
    fig, ax = plt.subplots()
    images = []

    while not done["__all__"]:
        actions = {}
        for agent in env.game.agent_pos.keys():
            actions[agent] = trainer.compute_action(
                observation=env.game.get_agents_obs()[agent],
                policy_id=agent,
            )
        env.step(actions)
        if record:
            im = env.game.render(fig, ax)
            images.append([im])

    if record:
        video_filename = f"{checkpoint_path}/video.mp4"
        ani = animation.ArtistAnimation(
            fig, images, interval=50, blit=True, repeat_delay=10000
        )
        ani.save(filename)
        print(f"Successfully wrote {filename}")

    # TODO print summary statistics


def main():
    args = ArgParser()
    config = load_config(args.config)
    env_config = config["env_config"]
    ray_config = config["ray_config"]
    run_config = config["run_config"]

    ray.shutdown()
    ray.init()
    register_env("ZSC-Cleaner", lambda _: CleanerEnv(env_config))

    results_dir = f"{os.path.expanduser('~')}/ray_results/{args.name}/"
    trainer = trainer_from_config(config, results_dir=results_dir)

    wandb.init(
        project=run_config["wandb_project"],
        entity=os.environ["USERNAME"],
        config=config,
        monitor_gym=True,
        sync_tensorboard=True,
        reinit=True,
    )  # integrate with Weights & Biases

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
