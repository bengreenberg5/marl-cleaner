from copy import deepcopy
import dill
import numpy as np
import os
import yaml
from collections import namedtuple
from gym.spaces import Box, Discrete
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.trainer import COMMON_CONFIG
from ray.tune.logger import UnifiedLogger


class Position(namedtuple("Position", ["i", "j"])):
    def __add__(self, other):
        if isinstance(other, Position):
            return Position(i=self.i + other.i, j=self.j + other.j)
        elif isinstance(other, int):
            return Position(i=self.i + other, j=self.j + other)
        elif isinstance(other, tuple):
            return Position(i=self.i + other[0], j=self.j + other[1])
        else:
            raise ValueError(
                "A Position can only be added to an int or another Position"
            )

    def __sub__(self, other):
        if isinstance(other, Position):
            return Position(i=self.i - other.i, j=self.j - other.j)
        elif isinstance(other, int):
            return Position(i=self.i - other, j=self.j - other)
        elif isinstance(other, tuple):
            return Position(i=self.i - other[0], j=self.j - other[1])
        else:
            raise ValueError(
                "A Position can only be added to an int or another Position"
            )

    def __eq__(self, other) -> bool:
        if isinstance(other, Position):
            return self.i == other.i and self.j == other.j
        if isinstance(other, (tuple, list)):
            assert (
                    len(other) == 2
            ), "Position equality comparison must be with a length-2 sequence"
            return self.i == other[0] and self.j == other[1]
        raise ValueError("A Position can only be compared with a Position-like item.")


MOVES = [
    Position(0, 0),  # NOOP
    Position(-1, 0),  # NORTH
    Position(1, 0),  # SOUTH
    Position(0, -1),  # WEST
    Position(0, 1),  # EAST
]
MASKS = {
    "clean": 0,
    "dirty": 1,
    "agent": 2,
    "wall": 3,
}


def grid_from_config(config):
    layout = config["env_config"]["layout"]
    return grid_from_layout(layout)


def grid_from_layout(layout):
    """
    Converts human-readable layout to grid format used internally by CleanerGame

    '''         {    clean:         dirty:         agent:         wall:
    XXXXX         [0 0 0 0 0]    [0 0 0 0 0]    [0 0 0 0 0]    [1 1 1 1 1]
    XADCX         [0 1 0 1 0]    [0 0 1 0 0]    [0 1 0 0 0]    [1 0 0 0 1]
    XDDAX   =>    [0 0 0 1 0]    [0 1 1 0 0]    [0 0 0 1 0]    [1 0 0 0 1]
    XXXXX         [0 0 0 0 0]    [0 0 0 0 0]    [0 0 0 0 0]    [1 1 1 1 1]
    '''         }
    """
    layout = np.array([list(line) for line in layout.rstrip("\n").split("\n")])
    height = len(layout)
    width = len(layout[0])
    grid = {mask: np.zeros((height, width)) for mask in MASKS.keys()}
    grid["clean"][np.where(layout == "C")] = 1
    grid["clean"][np.where(layout == "A")] = 1
    grid["dirty"][np.where(layout == "D")] = 1
    grid["agent"][np.where(layout == "A")] = 1
    grid["wall"][np.where(layout == "X")] = 1
    return grid


def agent_pos_from_grid(grid):
    """
    Returns a tuple of agent positions from the grid -- top to bottom, left to right
    """
    agent_pos = np.where(grid["agent"])
    return {
        f"a{num}": Position(agent_pos[0][num], agent_pos[1][num]) for num in range(len(agent_pos))
    }


def trainer_from_config(config):
    """
    Returns a trainer object from a dict of params
    """

    def policy_config(policy_name):
        if policy_name == "dqn":
            # return { "model": {
            #         "custom_options": config["model_config"],
            #     }}
            return {}
        raise NotImplemented(f"unknown policy {policy_name}")

    grid = grid_from_config(config)
    obs_dims = (len(grid["clean"]), len(grid["clean"][0]), 4)
    obs_space = Box(0, 1, obs_dims, dtype=np.int32)
    action_space = Discrete(5)
    policies = config["policy_config"]
    multi_agent_config = {
        "policies": {
            f"a{num}": (None, obs_space, action_space, policy_config(policy_name))
            for num, policy_name in config["policy_config"].items()
        },
        "policy_mapping_fn": lambda agent_id: agent_id,
    }
    model_config = {
        "dim": 3,
        "conv_filters": [
            [16, [2, 2], 1],
            [32, [5, 5], 1],
        ],
    }
    trainer_config = {
        "multiagent": multi_agent_config,
        "model": model_config,
        "env_config": config["env_config"],
        **config["ray_config"],
        # "callbacks" : TrainingCallbacks,
    }
    return DQNTrainer(
        trainer_config,
        "ZSC-Cleaner",
        logger_creator=lambda cfg: UnifiedLogger(cfg, "log"),
    )


def save_trainer(trainer, config, path=None):
    save_path = trainer.save(path)
    config = deepcopy(config)
    config_path = os.path.join(os.path.dirname(save_path), "config.pkl")
    with open(config_path, "wb") as f:
        dill.dump(config, f)
    return save_path


def load_config(config_name):
    fname = f"configs/{config_name}.yaml"
    try:
        with open(fname, "r") as f:
            config = yaml.safe_load(f.read())
            return config
    except FileNotFoundError:
        print(f"bad config path: {fname}")
