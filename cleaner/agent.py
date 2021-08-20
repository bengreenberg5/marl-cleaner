import numpy as np
import os
from copy import deepcopy

from gym.spaces import Box, Discrete
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.logger import UnifiedLogger

from cleaner.utils import RAY_DIR, obs_dims


class Agent(object):
    def __init__(self, policy_name):
        assert policy_name in ["ppo", "dqn"], f"unknown policy name: {policy_name}"
        self.policy_name = policy_name
        self.trainer = None

    def prepare_to_run(self, run_name, agent_num, config, checkpoint_num=None):
        """
        Note: if loading a checkpoint, env dimensions must match.
        Otherwise, the Trainer won't be able to interpret input from observations.
        """
        self.run_name = run_name
        self.agent_num = agent_num
        self.results_dir = f"{RAY_DIR}/{run_name}"
        self.name = f"{run_name}:{agent_num}"
        if checkpoint_num:
            checkpoint_path = f"{self.results_dir}/checkpoint_{str(checkpoint_num).zfill(6)}/checkpoint-{checkpoint_num}"
            agent_names = [f"{run_name}:{num}" for num in range(config["env_config"]["num_agents"])]
            agents = {agent_name: self for agent_name in agent_names}
            self.trainer = Agent.create_trainer(agents, self.policy_name, config, self.results_dir)
            self.trainer.restore(checkpoint_path)

    @staticmethod
    def create_trainer(agents, policy_name, config, results_dir):
        obs_space = Box(0, 1, obs_dims(config), dtype=np.int32)
        action_space = Discrete(5)
        policy = (None, obs_space, action_space, {})
        if config["run_config"]["heterogeneous"]:
            multi_agent_config = {
                "policies": {agent_name: deepcopy(policy) for agent_name in agents.keys()},
                "policy_mapping_fn": lambda agent_name: agent_name,
            }
        else:
            multi_agent_config = {
                "policies": {"agent_policy": policy},
                "policy_mapping_fn": lambda agent_name: "agent_policy",
            }
        model_config = {
            "conv_filters": [
                [16, [3, 3], 2],
                [32, [4, 4], 1],
            ],
            "conv_activation": "relu",
        }
        eval_config = {"verbose": config["run_config"]["verbose"]}
        trainer_config = {
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "multiagent": multi_agent_config,
            "model": model_config,
            "env_config": config["env_config"],
            "callbacks": DefaultCallbacks,
            "evaluation_config": eval_config,
            **config["ray_config"],
        }
        if policy_name == "dqn":
            trainer = DQNTrainer(
                trainer_config,
                "ZSC-Cleaner",
                logger_creator=lambda cfg: UnifiedLogger(cfg, results_dir),
            )
        else:
            trainer = PPOTrainer(
                trainer_config,
                "ZSC-Cleaner",
                logger_creator=lambda cfg: UnifiedLogger(cfg, results_dir),
            )
        for _, agent in agents.items():
            agent.trainer = trainer
        return trainer


