import numpy as np
import os
from copy import deepcopy

from gym.spaces import Box, Discrete
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.logger import UnifiedLogger

from cleaner.utils import RAY_DIR, obs_dims, create_trainer


class Agent(object):
    def __init__(self, policy_name, run_name, agent_num, config, seed, heterogeneous):
        assert policy_name in ["ppo", "dqn"], f"unknown policy name: {policy_name}"
        self.policy_name = policy_name
        self.run_name = run_name
        self.agent_num = agent_num
        self.config = config
        self.seed = seed
        self.heterogeneous = heterogeneous
        self.trainer = None
        self.results_dir = f"{RAY_DIR}/{run_name}"
        self.name = f"{run_name}:{agent_num}"
        self.eval_name = None

    # def restore_checkpoint(self, trainer, checkpoint_num):
    #     checkpoint_path = f"{self.results_dir}/checkpoint_{str(checkpoint_num).zfill(6)}/checkpoint-{checkpoint_num}"
    #     agents = {
    #         f"{self.run_name}:{num}": self
    #         for num in range(self.config["env_config"]["num_agents"])
    #     }
    #     self.trainer = create_trainer(
    #         self.policy_name,
    #         agents,
    #         self.config,
    #         self.results_dir,
    #         seed=1,
    #         heterogeneous=True,
    #     )
    #     self.trainer.load_checkpoint(checkpoint_path)
