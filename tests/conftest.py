import pytest

from voltvar_rl.config import load_config
from voltvar_rl.env import VoltVarEnv


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def env(cfg):
    return VoltVarEnv(cfg)
