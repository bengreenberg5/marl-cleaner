import pytest

from cleaner.utils import load_config
from cleaner.cleaner_env import CleanerEnv


@pytest.fixture
def env():
    config = load_config("simple_2")
    env = CleanerEnv(config["env_config"], "test", None)
    return env


def test_create(env):
    assert isinstance(env, CleanerEnv)
    env.game.validate_grid()


# TODO
