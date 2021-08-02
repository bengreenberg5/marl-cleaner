import gym
from ray.rllib.env import MultiAgentEnv

class CleanerEnv(MultiAgentEnv, gym.Env):
    
    def __init__(self, env_config):
        super().__init__()
        pass  # TODO
    
    def reset(self):
        pass  # TODO

    def step(self, actions):
        pass  # TODO

    def render(self):
        pass  # TODO
