from copy import deepcopy
import numpy as np

from cleaner.utils import *


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
        self.agent_pos = agent_pos_from_grid(self.grid)
        return self.agent_obs()

    def is_done(self):
        done = self.grid["clean"].sum().sum() == 0
        return {"__all__": done}

    def agent_obs(self):
        obs = np.stack([self.grid[layer] for layer in ["clean", "dirty", "agent", "wall"]], axis=-1)
        return {agent: obs for agent in self.agent_pos.keys()}

    def step(self, actions):
        reward = 0
        for agent, action in actions.items():
            new_pos = self.agent_pos[agent] + MOVES[action]
            if self.grid["wall"][new_pos]:
                continue
            if self.grid["dirty"][new_pos]:
                reward += 1
                self.grid["dirty"][new_pos] = 0
                self.grid["clean"][new_pos] = 1
                self.grid["agent"][new_pos] = 1
            self.agent_pos[agent] = new_pos
        return {agent: reward for agent in self.agent_pos.keys()}

    def render(self, fig=None, ax=None):
        pass  # TODO
