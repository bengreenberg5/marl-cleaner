from collections import namedtuple
from copy import deepcopy
from gym.spaces import Box, Discrete
import matplotlib
import numpy as np
import os
from typing import Dict, Optional
import yaml

from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer
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
    "wall": 2,
    "agent": 3,
}
COLORS = matplotlib.colors.ListedColormap(
    ["green", "red", "grey", "white"]  # clean (and no agent)  # dirty  # agent  # wall
)
RAY_DIR = f"{os.path.expanduser('~')}/ray_results"


def grid_from_config(config):
    env_config = config["env_config"]
    return grid_from_layout(env_config["layout"])


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
    num_agents = len(np.where(layout == "A")[0])
    grid["dirty"][np.where(layout == "D")] = 1
    grid["clean"][np.where(layout == "C")] = 1
    grid["wall"][np.where(layout == "X")] = 1
    start_pos_list = np.where(layout == "A")
    pos_list = [(start_pos_list[0][i], start_pos_list[1][i]) for i in range(num_agents)]
    for i, j in pos_list:
        grid["agent"][i][j] = 1
        grid["clean"][i][j] = 1
        grid["dirty"][i][j] = 0
    return grid


def grid_3d_to_2d(grid):
    """
    Squashes 4 layers into 1
    """
    board = np.zeros(grid["clean"].shape)
    board[np.where(grid["clean"])] = 0
    board[np.where(grid["dirty"])] = 1
    board[np.where(grid["wall"])] = 2
    board[np.where(grid["agent"])] = 3
    return board


def agent_pos_from_grid(grid, random_start=False):
    """
    Returns a tuple of agent positions from the grid -- top to bottom, left to right by default
    """
    agent_pos = np.where(grid["agent"])
    num_agents = len(agent_pos[0])
    agent_order = (
        np.random.permutation(num_agents) if random_start else range(num_agents)
    )
    return [Position(agent_pos[0][num], agent_pos[1][num]) for num in agent_order]


def obs_dims(config):
    grid = grid_from_config(config)
    dims = (len(grid["clean"]), len(grid["clean"][0]), 5)
    return dims


def create_trainer(
    policy_name,
    agents,
    config,
    results_dir,
    seed=1,
    heterogeneous=True,
    num_workers=None,
):
    obs_shape = obs_dims(config)
    obs_space = Box(0, 1, obs_shape, dtype=np.int32)
    action_space = Discrete(5)
    policy = (None, obs_space, action_space, {})
    if heterogeneous:
        multi_agent_config = {
            "policies": {agent_name: deepcopy(policy) for agent_name in agents.keys()},
            "policy_mapping_fn": lambda agent_name: agent_name,
        }
    else:
        multi_agent_config = {
            "policies": {"agent_policy": policy},
            "policy_mapping_fn": lambda agent_name: "agent_policy",
        }
    kernel_0_dim = [config["model_config"]["conv_kernel_size"]] * 2
    kernel_1_dim = list(obs_shape[:2])
    model_config = {
        "conv_filters": [
            [16, kernel_0_dim, 1],
            [32, kernel_1_dim, 1],
        ],
        "conv_activation": "relu",
    }
    eval_config = {"verbose": config["run_config"]["verbose"]}
    if num_workers:
        config = deepcopy(config)
        config["ray_config"]["num_workers"] = num_workers
    trainer_config = {
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "multiagent": multi_agent_config,
        "model": model_config,
        "env_config": config["env_config"],
        "callbacks": DefaultCallbacks,
        "evaluation_config": eval_config,
        "seed": seed,
        **config["ray_config"],
    }
    from pprint import pprint

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
    return trainer


def save_trainer(trainer, path=None, verbose=True):
    save_path = trainer.save(path)
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
