from collections import defaultdict
from matplotlib import animation
import os
from pathlib import Path
import stat
from typarse import BaseParser
from typing import Dict, Optional
import wandb

import ray
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.typing import PolicyID
from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.agents import DefaultCallbacks
from ray.tune import register_env

from cleaner.cleaner_env import *


class ArgParser(BaseParser):
    config: str = "simple_3x3"
    name: str = "cleaner"
    training_iters: int = 5
    checkpoint_freq: int = 25
    eval_freq: int = 25

    _help = {
        "config":           "Path to the config of the experiment",
        "name":             "Name of subdirectory containing results for this experiment",
        "training_iters":   "Number of training iterations",
        "checkpoint_freq":  "How many training iterations between checkpoints; a value of 0 (default) disables checkpointing",
        "eval_freq":        "How many training iterations between evaluations"
    }


def evaluate(eval_config, record=True):
    """
    eval_config = {
        "agents": [  # run_name, agent_num, checkpoint_num
            ("ppo5", 0, 1000),
            ("ppo5", 1, 1000),
        ],
        "env_config": {
            layout: '''
                    XXXXXXX
                    XADDDDX
                    XDDDDDX
                    XDDDDDX
                    XDDDDDX
                    XDDDDAX
                    XXXXXXX
                    ''',
            tick_limit: 12,
        },
        "eval_name": "my_eval",
    }
    """
    agents = {}
    trainers = {}
    for run_name, agent_num, checkpoint_num in agents:
        agent = Agent(run_name, agent_num)
        agents[agent.name] = agent
        trainer = agent.get_trainer(checkpoint_num)
        trainers[agent.name] = trainer

    done = {"__all__": False}
    env = CleanerEnv(eval_config["env_config"], run_name=eval_config["eval_name"], agent_names=list(agents.keys()))
    fig, ax = plt.subplots()
    images = []

    # run episode
    rewards = []
    actions = {}
    while not done["__all__"]:
        if record:
            im = env.game.render(fig, ax)
            images.append([im])
        for agent_name in agents.keys():
            actions[agent_name] = trainers[agent_name].compute_action(
                observation=env.game.agent_obs()[agent_name],
                policy_id=agent_name,
            )
        _, reward, done, _ = env.step(actions)
        rewards.append(reward)
    print(f"episode reward: {sum(rewards)}")

    if record:
        video_filename = f"{RAY_DIR}/{eval_config['eval_name']}/video.mp4"
        ani = animation.ArtistAnimation(
            fig, images, interval=200, blit=True, repeat_delay=10000
        )
        ani.save(video_filename)
        print(f"saved video at {video_filename}")


def main():
    args = ArgParser()
    config = load_config(args.config)
    env_config = config["env_config"]
    ray_config = config["ray_config"]
    run_config = config["run_config"]

    ray.shutdown()
    ray.init()
    register_env("ZSC-Cleaner", lambda _: CleanerEnv(env_config, run_name=args.name))

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
        trainer.train()
        if args.checkpoint_freq != 0 and i % args.checkpoint_freq == 0:
            save_trainer(trainer, config, path=results_dir, verbose=verbose)
        if args.eval_freq != 0 and i % args.eval_freq == 0:
            evaluate(config, args.name, i+1, record=True)

    save_trainer(trainer, config, path=results_dir, verbose=verbose)
    evaluate(config, args.name, args.training_iters, record=True)

    ray.shutdown()


if __name__ == "__main__":
    main()
