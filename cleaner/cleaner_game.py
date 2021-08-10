from collections import namedtuple
from copy import deepcopy
import numpy as np

from cleaner.utils import *


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


class CleanerGame:
    def __init__(self, layout):
        self.layout = layout  # human-readable layout
        self.size = (len(layout), len(layout[0]))  # height and width of grid
        self.grid = grid_from_layout(layout)  # masks for environment features
        self.agent_pos = agent_pos_from_grid(self.grid)  # tuples of agent positions
        self.num_agents = len(self.agent_pos)  # number of agents in environment
        self._validate_grid()

    def __repr__(self):
        return self.layout

    def _validate_grid(self):
        clean_dirty_wall = self.grid["clean"] + self.grid["dirty"] + self.grid["wall"]
        agent_dirty_wall = self.grid["agent"] + self.grid["dirty"] + self.grid["wall"]
        clean_agent = self.grid["agent"] + self.grid["clean"]
        assert (
            clean_dirty_wall.max() == 1
        ), "position cannot contain more than one of ('clean', 'dirty', 'wall')"
        assert (
            agent_dirty_wall.max() == 1
        ), "position cannot contain more than one of ('agent', 'dirty', 'wall')"
        assert (
            np.count_nonzero(clean_agent == 1) == 0
        ), "position containing agent must be clean"

    def reset(self):
        self.grid = grid_from_layout(self.layout)

    def is_done(self):
        return self.grid["clean"].sum().sum() == 0

    def step(self, actions):
        reward = 0
        for agent, action in enumerate(actions):
            new_pos = self.agent_pos[agent] + MOVES[action]
            if self.grid["wall"][new_pos]:
                continue
            if self.grid["dirty"][new_pos]:
                reward += 1
                self.grid["dirty"][new_pos] = 0
                self.grid["clean"][new_pos] = 1
                self.grid["agent"][new_pos] = 1
            self.agent_pos[agent] = new_pos
        return reward

    def render(self, fig=None, ax=None):
        pass  # TODO
