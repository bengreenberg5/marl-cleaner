from collections import namedtuple
from copy import deepcopy
import dill
from gym.spaces import Box, Discrete
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Optional
import yaml

from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.trainer import COMMON_CONFIG
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.typing import PolicyID
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


class CleanerCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()

    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: MultiAgentEpisode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        episode.user_data["rewards"] = list()

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        rewards = {}
        for agent in base_env.get_unwrapped()[0].game.agent_pos.keys():
            rewards[agent] = episode.prev_reward_for(agent)
        episode.user_data["rewards"].append(rewards)

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: MultiAgentEpisode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        # custom metrics get saved to the logfile
        episode.custom_metrics["rewards"] = sum(
            [sum(list(rewards.values())) for rewards in episode.user_data["rewards"]]
        )


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
COLORS = matplotlib.colors.ListedColormap(
    ["green", "red", "white", "grey"]  # clean (and no agent)  # dirty  # agent  # wall
)
RAY_DIR = f"{os.path.expanduser('~')}/ray_results"


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
    layout = np.array(
        [
            list(line)
            for line in layout.replace(" ", "").lstrip("\n").rstrip("\n").split("\n")
        ]
    )
    height = len(layout)
    width = len(layout[0])
    grid = {mask: np.zeros((height, width)) for mask in MASKS.keys()}
    grid["clean"][np.where(layout == "C")] = 1
    grid["clean"][np.where(layout == "A")] = 1
    grid["dirty"][np.where(layout == "D")] = 1
    grid["agent"][np.where(layout == "A")] = 1
    grid["wall"][np.where(layout == "X")] = 1
    return grid


def grid_3d_to_2d(grid):
    """
    Squashes 4 layers into 1
    """
    board = np.zeros(grid["clean"].shape)
    board[np.where(grid["clean"])] = 0
    board[np.where(grid["dirty"])] = 1
    board[np.where(grid["agent"])] = 2
    board[np.where(grid["wall"])] = 3
    return board


def agent_pos_from_grid(grid):
    """
    Returns a tuple of agent positions from the grid -- top to bottom, left to right
    """
    agent_pos = np.where(grid["agent"])
    return [
        Position(agent_pos[0][num], agent_pos[1][num]) for num in range(len(agent_pos))
    ]


def trainer_from_config(config, results_dir):
    """
    Returns a trainer object from a dict of params
    """

    def policy_config(policy_name):
        if policy_name == "dqn":
            # return {"model": {"custom_options": config["model_config"]}}
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
            [16, [3, 3], 2],
            [32, [4, 4], 1],
        ],
        "conv_activation": "relu",
    }
    eval_config = {"verbose": True}
    trainer_config = {
        "multiagent": multi_agent_config,
        # "model": model_config,
        "env_config": config["env_config"],
        "callbacks": DefaultCallbacks,
        "evaluation_config": eval_config,
        **config["ray_config"],
    }

    return PPOTrainer(
        trainer_config,
        "ZSC-Cleaner",
        logger_creator=lambda cfg: UnifiedLogger(cfg, results_dir),
    )


def save_trainer(trainer, config, path=None, verbose=True):
    save_path = trainer.save(path)
    config = deepcopy(config)
    config_path = os.path.join(os.path.dirname(save_path), "config.pkl")
    with open(config_path, "wb") as f:
        dill.dump(config, f)
    if verbose:
        print(f"saved trainer at {save_path}")
    return save_path


def load_config(config_name):
    fname = f"configs/{config_name}.yaml"
    try:
        with open(fname, "r") as f:
            config = yaml.safe_load(f.read())
            return config
    except FileNotFoundError:
        print(f"bad config path: {fname}")
