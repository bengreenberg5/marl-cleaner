import gym
from ray.rllib.env import MultiAgentEnv

from cleaner.cleaner_game import *


class CleanerEnv(MultiAgentEnv, gym.Env):
    def __init__(
        self,
        env_config: Dict[str, Any],
        run_name: str,
        agent_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.env_config = env_config
        self.run_name = run_name
        if not agent_names:
            agent_names = [
                f"{run_name}:{num}" for num in range(env_config["num_agents"])
            ]
        self.agent_names = agent_names
        self.game = CleanerGame(**env_config, agent_names=agent_names)
        self.observation_space = Box(
            0, 1, obs_dims({"env_config": env_config}), dtype=np.int32
        )
        self.action_space = Discrete(5)

    def reset(self):
        grid = self.game.reset()
        return grid

    def step(self, actions: Dict[str, int]):
        reward = self.game.step(actions)
        done = self.game.is_done()
        info = {agent: {} for agent in self.agent_names}
        return self.game.get_agent_obs(), reward, done, info

    def render(self, fig=None, ax=None, mode=None):
        return self.game.render(fig, ax)
