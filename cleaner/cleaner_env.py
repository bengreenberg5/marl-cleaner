import gym
from gym.spaces import Box, Discrete
from ray.rllib.env import MultiAgentEnv

from cleaner.cleaner_game import *


class CleanerEnv(MultiAgentEnv, gym.Env):
    def __init__(self, env_config, run_name):
        super().__init__()
        if "agent_names" not in env_config:
            env_config["agent_names"] = [
                f"{run_name}:{num}" for num in range(self.game.num_agents)
            ]
        self.game = CleanerGame(**env_config)
        self.observation_space = Box(
            0, 1, obs_dims({"env_config": env_config}), dtype=np.int32
        )
        self.action_space = Discrete(5)

    def reset(self):
        grid = self.game.reset()
        return grid

    def step(self, actions):
        reward = self.game.step(actions)
        done = self.game.is_done()
        info = {agent: {} for agent in self.agent_names.keys()}
        return self.game.agent_obs(), reward, done, info

    def render(self, fig=None, ax=None, mode=None):
        return self.game.render(fig, ax)
