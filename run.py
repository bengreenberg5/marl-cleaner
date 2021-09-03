import matplotlib.pyplot as plt
from matplotlib import animation
from typarse import BaseParser

import ray
from ray.tune import register_env

from cleaner.agent import Agent
from cleaner.cleaner_env import *


class ArgParser(BaseParser):
    name: str = "cleaner_run"
    config: str = "simple_5x5"
    policy: str = "ppo"
    training_iters: int = 5
    seed: int = 1
    homogeneous: bool = False
    random_start: bool = False
    checkpoint_freq: int = 25
    eval_freq: int = 25

    _help = {
        "config": "Path to the config of the experiment",
        "name": "Name of subdirectory containing results for this experiment",
        "policy": "RL algorithm",
        "training_iters": "Number of training iterations",
        "seed": "Random seed for Ray workers",
        "homogeneous": "Centrally train one policy for all agents",
        "random_start": "Randomly initialize the starting positions",
        "checkpoint_freq": "How many training iterations between trainer checkpoints",
        "eval_freq": "How many training iterations between evaluations",
    }


def evaluate(
    agents,
    eval_config,
    eval_run_name,
    seed=1,
    heterogeneous=True,
    num_episodes=1,
    video_filename=None,
    record=True,
):
    fig, ax = plt.subplots()
    images = []
    ep_rewards = []
    agent_names = [agent.name for agent in agents.values()]

    for ep in range(num_episodes):
        env = CleanerEnv(
            eval_config["env_config"], run_name=eval_run_name, agent_names=agent_names
        )
        ep_reward = 0
        actions = {}
        done = {"__all__": False}

        while not done["__all__"]:
            if ep == num_episodes - 1 and record:
                im = env.game.render(fig, ax)
                images.append([im])
            obs = env.game.get_agent_obs()
            for agent_name, agent in agents.items():
                policy_id = agent_name if heterogeneous else "agent_policy"
                actions[agent_name] = agent.trainer.compute_action(
                    observation=obs[agent_name],
                    policy_id=policy_id,
                )
            _, reward, done, _ = env.step(actions)
            ep_reward += sum(list(reward.values()))
        ep_rewards.append(ep_reward)

    # create video from last episode
    if record:
        results_dir = f"{RAY_DIR}/{eval_run_name}"
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        if not video_filename:
            video_filename = f"{results_dir}/video.mp4"
        ani = animation.ArtistAnimation(
            fig, images, interval=100, blit=True, repeat_delay=1000
        )
        ani.save(video_filename)
        print(f"saved video at {video_filename}")
    else:
        ani = None

    print(f"episode rewards: {ep_rewards} (mean = {sum(ep_rewards) / len(ep_rewards)})")
    return ep_rewards, ani


def train(
    run_name,
    config,
    policy_name,
    training_iters,
    seed,
    heterogeneous,
    checkpoint_freq=0,
    eval_freq=0,
    num_eval_episodes=5,
    verbose=True,
):
    # initialize agents
    agents = {}
    for agent_num in range(config["env_config"]["num_agents"]):
        agent = Agent(policy_name, run_name, agent_num, config, seed, heterogeneous)
        agents[agent.name] = agent

    # run training
    results_dir = list(agents.values())[0].results_dir
    trainer = create_trainer(
        policy_name, agents, config, results_dir, seed=seed, heterogeneous=heterogeneous
    )
    for i in range(training_iters):
        if verbose:
            print(f"starting training iteration {i}")
        trainer.train()
        if checkpoint_freq != 0 and i % checkpoint_freq == 0:
            save_trainer(trainer, path=results_dir, verbose=verbose)
        if eval_freq != 0 and i % eval_freq == 0:
            video_filename = f"{results_dir}/checkpoint_{str(i+1).zfill(6)}/video.mp4"
            for agent in agents.values():
                agent.trainer = trainer
            evaluate(
                agents=agents,
                eval_config=config,
                eval_run_name=run_name,
                seed=seed,
                heterogeneous=heterogeneous,
                video_filename=video_filename,
                num_episodes=num_eval_episodes,
                record=True,
            )
    save_trainer(trainer, path=results_dir, verbose=verbose)
    video_filename = (
        f"{results_dir}/checkpoint_{str(training_iters).zfill(6)}/video.mp4"
    )
    evaluate(
        agents=agents,
        eval_config=config,
        eval_run_name=run_name,
        seed=seed,
        heterogeneous=heterogeneous,
        video_filename=video_filename,
        num_episodes=num_eval_episodes,
        record=True,
    )
    return agents


def main():
    args = ArgParser()
    config = load_config(args.config)
    config["env_config"]["random_start"] = args.random_start  # hacky

    # initialize ray
    ray.shutdown()
    ray.init()
    register_env("ZSC-Cleaner", lambda _: CleanerEnv(config["env_config"], run_name=args.name))

    # train model(s)
    train(
        run_name=args.name,
        config=config,
        policy_name=args.policy,
        training_iters=args.training_iters,
        seed=args.seed,
        heterogeneous=not args.homogeneous,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        num_eval_episodes=5,
        verbose=config["run_config"]["verbose"],
    )
    ray.shutdown()
    print(f"finished training {args.name}")


if __name__ == "__main__":
    main()
