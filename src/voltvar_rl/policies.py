"""Control strategies: fixed-curve baselines and the trained PPO policy.

Every strategy exposes ``curves(env, obs) -> (vv, vw)``. All of them run through the same
environment, scenarios, inverter model, measurement and reward.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import curves as C
from .env import VoltVarEnv
from .ppo import PPO


class FixedCurves:
    def __init__(self, name: str, make: Callable[[int, float], tuple[np.ndarray, np.ndarray]]):
        self.name = name
        self._make = make

    def curves(self, env: VoltVarEnv, obs: np.ndarray):
        return self._make(env.n_pv, env.cfg.pv.q_max_pu)


class PPOPolicy:
    def __init__(self, agent: PPO, name: str = "PPO", deterministic: bool = True):
        self.name = name
        self.agent = agent
        self.deterministic = deterministic

    def curves(self, env: VoltVarEnv, obs: np.ndarray):
        action, _, _ = self.agent.act(obs, deterministic=self.deterministic)
        return env.action_to_curves(action)


def baselines() -> list[FixedCurves]:
    return [
        FixedCurves("No control", lambda n, q: (C.vv_disabled(n), C.vw_disabled(n))),
        FixedCurves("Volt-VAR", lambda n, q: (C.vv_ieee1547(n, q), C.vw_disabled(n))),
        FixedCurves("Volt-Watt", lambda n, q: (C.vv_disabled(n), C.vw_ieee1547(n))),
        FixedCurves("VV+VW", lambda n, q: (C.vv_ieee1547(n, q), C.vw_ieee1547(n))),
        FixedCurves("VV+VW tight", lambda n, q: (C.vv_aggressive(n, q), C.vw_aggressive(n))),
    ]
