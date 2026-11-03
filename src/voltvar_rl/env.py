"""Reinforcement-learning environment: PPO picks Volt/VAR + Volt/Watt curves for every inverter.

Each step:

1. The policy outputs new curves for all inverters (or a baseline supplies fixed curves).
2. Each inverter reads its local voltage from the previous power flow, looks up its target
   reactive power and active-power ceiling on its curves, and moves a fraction
   ``env.damping`` of the way toward them. This is the same damped quasi-static update a
   real inverter (or OpenDSS InvControl) performs.
3. The power flow is solved again at the same hour, and the reward is computed from the
   new voltages and the measured PV output.

Repeated steps act as a fixed-point iteration of the inverter control loop. Episodes have a
fixed length (no early termination).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import curves as C
from .config import Config
from .feeder import Feeder
from .profiles import make_day
from .reward import compute_reward


@dataclass
class StepInfo:
    reward: float
    v_min: float
    v_max: float
    n_violations: int
    p_kw: float
    p_available_kw: float
    q_abs_kvar: float

    @property
    def curtailment_pct(self) -> float:
        if self.p_available_kw <= 1e-6:
            return 0.0
        return 100.0 * max(0.0, 1.0 - self.p_kw / self.p_available_kw)


class VoltVarEnv:
    def __init__(self, cfg: Config, feeder: Feeder | None = None):
        self.cfg = cfg
        self.feeder = feeder or Feeder(cfg)
        self.n_pv = self.feeder.n_pv
        self.obs_dim = len(self.feeder.node_names)
        self.act_dim = self.n_pv * C.PARAMS_PER_PV
        self.band = tuple(cfg.feeder.voltage_band)
        self.fleet_pmpp = self.n_pv * cfg.pv.pmpp
        self.fleet_q = self.n_pv * cfg.pv.q_max_pu * cfg.pv.kva
        self.day: int | None = None
        self.hour: int | None = None
        self.t = 0
        self.v = np.ones(self.obs_dim)
        self.vv = C.vv_disabled(self.n_pv)
        self.vw = C.vw_disabled(self.n_pv)

    # ---------------------------------------------------------------- helpers
    def observation(self) -> np.ndarray:
        """Node voltages, centred on 1 pu and scaled so the ANSI band maps to [-1, 1]."""
        return ((self.v - 1.0) / 0.05).astype(np.float32)

    def _measure(self) -> StepInfo:
        p, q = self.feeder.pv_powers()
        r = compute_reward(self.v, p, q, fleet_pmpp_kw=self.fleet_pmpp, fleet_q_kvar=self.fleet_q,
                           band=self.band, cfg=self.cfg.reward)
        lo, hi = self.band
        return StepInfo(
            reward=r,
            v_min=float(self.v.min()),
            v_max=float(self.v.max()),
            n_violations=int(((self.v < lo) | (self.v > hi)).sum()),
            p_kw=float(p.sum()),
            p_available_kw=float(self.feeder.available_power(self.hour).sum()),
            q_abs_kvar=float(np.abs(q).sum()),
        )

    # -------------------------------------------------------------------- API
    def reset(self, day: int, hour: int) -> np.ndarray:
        self.day, self.hour, self.t = int(day), int(hour), 0
        self.feeder.set_profiles(make_day(self.cfg.profiles, self.cfg.scenario_seed, self.day, self.n_pv))
        self.feeder.reset_inverters()
        self.feeder.solve_at(self.hour)
        self.v = self.feeder.node_voltages()
        self.vv = C.vv_disabled(self.n_pv)
        self.vw = C.vw_disabled(self.n_pv)
        self.p_set = self.feeder.available_power(self.hour).copy()
        self.q_set = np.zeros(self.n_pv)
        return self.observation()

    def initial_info(self) -> StepInfo:
        """Metrics of the uncontrolled state right after ``reset``."""
        return self._measure()

    def action_to_curves(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return C.action_to_curves(action, self.n_pv, self.cfg.curves, self.cfg.pv.q_max_pu)

    def step(self, action: np.ndarray):
        return self.step_curves(*self.action_to_curves(action))

    def step_curves(self, vv: np.ndarray, vw: np.ndarray):
        """Apply curves (n_pv, 2, 5) for one control iteration.

        Returns ``(obs, reward, truncated, info)``.
        """
        self.vv, self.vw = vv, vw
        v_local = self.v[self.feeder.pv_node_idx]
        q_pu = C.evaluate(vv, v_local)
        p_pu = np.clip(C.evaluate(vw, v_local), 0.0, 1.0)

        kva = self.cfg.pv.kva
        p_limit = p_pu * self.feeder.available_power(self.hour)
        q_headroom = np.sqrt(np.maximum(kva ** 2 - p_limit ** 2, 0.0))
        q = np.clip(q_pu * kva, -q_headroom, q_headroom)

        # First-order response toward the curve target (like OpenDSS InvControl's
        # deltaQ_factor / deltaP_factor). Without it, steep curves oscillate.
        a = self.cfg.env.damping
        self.p_set = self.p_set + a * (p_limit - self.p_set)
        self.q_set = self.q_set + a * (q - self.q_set)
        self.feeder.set_pv_setpoints(self.p_set, self.q_set)
        self.feeder.solve_at(self.hour)
        self.v = self.feeder.node_voltages()
        self.t += 1
        info = self._measure()
        truncated = self.t >= self.cfg.env.steps_per_episode
        return self.observation(), info.reward, truncated, info
