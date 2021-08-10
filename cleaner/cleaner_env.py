import gym
from gym.spaces import Box, Discrete
from ray.rllib.env import MultiAgentEnv

from cleaner.cleaner_game import *


class CleanerEnv(MultiAgentEnv, gym.Env):
    def __init__(self, env_config):
        super().__init__()
        self.game = CleanerGame(**env_config)
        self.observation_space = Box(0, 1, dtype=np.int32)
        self.action_space = Discrete(5)

    def reset(self):
        self.game.reset()

    def step(self, actions):
        reward = self.game.step(actions)
        info = ""
        done = self.game.is_done()
        return self.game.grid, reward, done, info

    def render(self, mode=None):
        pass  # TODO
