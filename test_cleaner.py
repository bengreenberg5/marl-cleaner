import pytest

from cleaner.utils import load_config, MASKS
from cleaner.cleaner_env import CleanerEnv


@pytest.fixture
def env():
    config = load_config("simple_2")
    env = CleanerEnv(config["env_config"], "test", None)
    print(env.game.get_agent_obs())
    return env


def test_create(env):
    assert isinstance(env, CleanerEnv)
    env.game.validate_grid()
    for layer in MASKS.keys():
        assert layer in env.game.grid


def test_step(env):
    assert env.game.tick == 0
    assert env.game.agent_pos["test:0"] == (1, 1)
    assert env.game.agent_pos["test:1"] == (11, 11)
    env.step({
        "test:0": 4,  # east
        "test:1": 3,  # west
    })
    assert env.game.tick == 1
    assert env.game.agent_pos["test:0"] == (1, 2)
    assert env.game.agent_pos["test:1"] == (11, 10)
    env.step({
        "test:0": 2,  # south
        "test:1": 1,  # north
    })
    assert env.game.tick == 2
    assert env.game.agent_pos["test:0"] == (2, 2)
    assert env.game.agent_pos["test:1"] == (10, 10)
    env.step({
        "test:0": 3,  # west
        "test:1": 4,  # east
    })
    assert env.game.tick == 3
    assert env.game.agent_pos["test:0"] == (2, 1)
    assert env.game.agent_pos["test:1"] == (10, 11)
    env.reset()
    assert env.game.tick == 0
    assert env.game.agent_pos["test:0"] == (1, 1)
    assert env.game.agent_pos["test:1"] == (11, 11)


def test_boundaries(env):
    actions = [{"test:0": i, "test:1": i} for i in range(5)]
    for action in actions:
        for i in range(100):
            env.step(action)


def test_cleaning(env):
    assert env.game.grid["clean"].sum().sum() == 2
    _, reward, _, _ = env.step({"test:0": 4, "test:1": 0})
    assert sum(reward.values()) == 0.75
    assert env.game.grid["clean"].sum().sum() == 3
    _, reward, _, _ = env.step({"test:0": 0, "test:1": 3})
    assert sum(reward.values()) == 0.75
    assert env.game.grid["clean"].sum().sum() == 4
    _, reward, _, _, = env.step({"test:0": 0, "test:1": 0})
    assert sum(reward.values()) == -0.25
    assert env.game.grid["clean"].sum().sum() == 4
    _, reward, _, _ = env.step({"test:0": 4, "test:1": 3})
    assert sum(reward.values()) == 1.75
    assert env.game.grid["clean"].sum().sum() == 6






























