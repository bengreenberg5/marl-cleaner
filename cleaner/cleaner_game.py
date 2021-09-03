from copy import deepcopy
import numpy as np

from cleaner.utils import *


class CleanerGame:
    def __init__(
        self,
        layout,
        tick_limit,
        num_agents,
        agent_names,
        time_penalty=-0.25,
        clean_penalty=0.0,
        random_start=False,
    ):
        self.layout = layout  # human-readable layout
        self.tick_limit = tick_limit  # how many time steps before game ends
        self.num_agents = num_agents  # how many agents on the board
        self.agent_names = agent_names  # list of agent identifiers
        self.time_penalty = time_penalty  # negative reward for each timestep
        self.clean_penalty = (
            clean_penalty  # negative reward for moving into a clean square
        )
        self.random_start = random_start
        self.size = (len(layout), len(layout[0]))  # height and width of grid
        self.reset()

    def __repr__(self):
        return self.layout

    def validate_grid(self):
        clean_dirty_wall = self.grid["clean"] + self.grid["dirty"] + self.grid["wall"]
        agent_dirty_wall = self.grid["agent"] + self.grid["dirty"] + self.grid["wall"]
        assert (
            clean_dirty_wall.max() == 1
        ), "position cannot contain more than one of ('clean', 'dirty', 'wall')"
        assert (
            agent_dirty_wall.max() == 1
        ), "position cannot contain more than one of ('agent', 'dirty', 'wall')"
        assert (
            self.grid["agent"].sum().sum() == self.num_agents
        ), "environment layout must correspond to `num_agents`"

    def reset(self):
        self.grid = grid_from_layout(self.layout)
        pos_list = agent_pos_from_grid(self.grid, random_start=self.random_start)
        self.agent_pos = {
            self.agent_names[i]: pos_list[i] for i in range(self.num_agents)
        }
        self.tick = 0
        self.validate_grid()
        return self.get_agent_obs()

    def is_done(self):
        done = self.tick == self.tick_limit or self.grid["dirty"].sum().sum() == 0
        return {"__all__": done}

    def get_agent_obs(self):
        layers = [self.grid[layer] for layer in ["clean", "dirty", "wall"]]
        layers.append(np.zeros(layers[0].shape))  # self
        layers.append(np.zeros(layers[0].shape))  # other
        base_obs = np.stack(layers, axis=-1)
        obs = {}
        for agent_name, agent_pos in self.agent_pos.items():
            agent_obs = deepcopy(base_obs)
            agent_obs[agent_pos][3] = 1
            for other_name, other_pos in self.agent_pos.items():
                if other_name != agent_name:
                    agent_obs[other_pos][4] = 1
            obs[agent_name] = agent_obs
        return obs

    def step(self, actions):
        reward = self.time_penalty
        for agent, action in actions.items():
            pos = self.agent_pos[agent]
            new_pos = pos + MOVES[action]
            if self.grid["wall"][new_pos]:
                continue
            self.grid["agent"][pos] = 0
            self.grid["agent"][new_pos] = 1
            if self.grid["clean"][new_pos] and action != 0:
                reward += self.clean_penalty
            elif self.grid["dirty"][new_pos]:
                self.grid["dirty"][new_pos] = 0
                self.grid["clean"][new_pos] = 1
                reward += 1
            self.agent_pos[agent] = new_pos
        self.tick += 1
        return {agent: reward / self.num_agents for agent in self.agent_names}

    def render(self, fig=None, ax=None):
        if not fig or not ax:
            fig, ax = plt.subplots()
        board = grid_3d_to_2d(self.grid)
        # board[0, 0] = 4
        im = ax.imshow(board, cmap=COLORS)
        return im
