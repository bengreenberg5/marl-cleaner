from copy import deepcopy
import dill
import numpy as np
import os
import yaml
from gym.spaces import Box, Discrete

from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.trainer import COMMON_CONFIG
from ray.tune.logger import UnifiedLogger


MOVES = [
    ( 0,  0),  # NOOP
    (-1,  0),  # NORTH
    ( 1,  0),  # SOUTH
    ( 0, -1),  # WEST
    ( 0,  1),  # EAST
]
MASKS = {
    "clean": 0,
    "dirty": 1,
    "agent": 2,
    "wall":  3,
}


def grid_from_config(config):
    """
    Converts human-readable layout to grid format used internally by CleanerGame

    '''         {    clean:         dirty:         agent:         wall:
    XXXXX         [0 0 0 0 0]    [0 0 0 0 0]    [0 0 0 0 0]    [1 1 1 1 1]
    XADCX         [0 1 0 1 0]    [0 0 1 0 0]    [0 1 0 0 0]    [1 0 0 0 1]
    XDDAX   =>    [0 0 0 1 0]    [0 1 1 0 0]    [0 0 0 1 0]    [1 0 0 0 1]
    XXXXX         [0 0 0 0 0]    [0 0 0 0 0]    [0 0 0 0 0]    [1 1 1 1 1]
    '''         }
    """
    layout = config["env_config"]["layout"]
    layout = [list(line) for line in layout.split("\n")]
    height = len(layout)
    width = len(layout[0])
    grid = { mask: np.zeros((height, width)) for mask in MASKS.keys()}
    grid["clean"][np.where(layout == "C")] = 1
    grid["dirty"][np.where(layout == "D")] = 1
    grid["agent"][np.where(layout == "A")] = 1
    grid["wall"][ np.where(layout == "X")] = 1
    return grid


def agent_pos_from_grid(grid):
    """
    Returns a tuple of agent positions from the grid -- top to bottom, left to right
    """
    agent_pos = np.where(grid["agent"])
    return [(agent_pos[0][i], agent_pos[1][i]) for i in range(len(agent_pos))]  # array of agent positions


def trainer_from_config(config):
    """
    Returns a trainer object from a dict of params
    """
    def policy_config(policy_name):
        if policy_name == "dqn":
            return {
                "model": {
                    "custom_options": config["model_config"],
                    # "custom_model": "MyPPOModel"
                }
            }
        raise NotImplemented(f"unknown policy {policy_name}")

    obs_space = Box(0, 1, np.int32)
    action_space = Discrete(5)
    policies = config["policy_config"]
    multi_agent_config = {
        "policies": (None, obs_space, action_space, policy_config(policy_name)
                     for _, policy_name in config["policy_config"].items()),
        # multi_agent_config['policy_mapping_fn'] = select_policy
        # multi_agent_config['policies_to_train'] = 'ppo'
    }
    trainer_config = {
        "multiagent": multi_agent_config,
        "env_config" : config["env_config"],
        **config["ray_config"],
        # "callbacks" : TrainingCallbacks,
    }
    return DQNTrainer(trainer_config, "ZSC-Cleaner", logger_creator=lambda cfg: UnifiedLogger(cfg, "log"))


def save_trainer(trainer, config, path=None):
    save_path = trainer.save(path)
    config = deepcopy(config)
    config_path = os.path.join(os.path.dirname(save_path), "config.pkl")
    with open(config_path, "wb") as f:
        dill.dump(config, f)
    return save_path


def load_config(config_name):
    try:
        fname = f"configs/{config_name}.yaml"
        with open(fname, "r") as f:
            config = yaml.safe_load(f.read())
            return config
    except FileNotFoundError:
        print(f"bad config path: {fname}")
















