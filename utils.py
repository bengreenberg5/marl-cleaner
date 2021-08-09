from copy import deepcopy
import dill
import numpy as np
import os
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


def grid_from_layout(layout):
    """
    Converts human-readable layout to grid format used internally by CleanerGame

    '''        {    clean:         dirty:         agent:         wall:
    XXXXX        [0 0 0 0 0]    [0 0 0 0 0]    [0 0 0 0 0]    [1 1 1 1 1]
    XADCX        [0 1 0 1 0]    [0 0 1 0 0]    [0 1 0 0 0]    [1 0 0 0 1]
    XDDAX  =>    [0 0 0 1 0]    [0 1 1 0 0]    [0 0 0 1 0]    [1 0 0 0 1]
    XXXXX        [0 0 0 0 0]    [0 0 0 0 0]    [0 0 0 0 0]    [1 1 1 1 1]
    '''        }
    """
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
    config = deepcopy(config)
    config["multiagent"] = None  # TODO define DQN policy here
    return DQNTrainer(config, "ZSC-Cleaner", logger_creator=lambda cfg: UnifiedLogger(cfg, "log"))


def save_trainer(trainer, config, path=None):
    save_path = trainer.save(path)
    config = deepcopy(config)
    config_path = os.path.join(os.path.dirname(save_path), "config.pkl")
    with open(config_path, "wb") as f:
        dill.dump(config, f)
    return save_path
















