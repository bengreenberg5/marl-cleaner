import gym
from gym.spaces import Box, Discrete
from ray.rllib.env import MultiAgentEnv

from cleaner.cleaner_game import *


class CleanerEnv(MultiAgentEnv, gym.Env):
    def __init__(self, env_config):
        super().__init__()
        self.game = CleanerGame(**env_config)
        self.agents = [f"a{num}" for num in range(self.game.num_agents)]
        self.observation_space = Box(0, 1, self.game.agent_obs()["a0"].shape, dtype=np.int32)
        self.action_space = Discrete(5)

    def reset(self):
        grid = self.game.reset()
        return grid

    def step(self, actions):
        agents = self.game.agent_pos.keys()
        reward = self.game.step(actions)
        done = self.game.is_done()
        info = {agent: {} for agent in agents}
        return self.game.agent_obs(), reward, done, info

    def render(self, fig=None, ax=None, mode=None):
        return self.game.render(fig, ax)
