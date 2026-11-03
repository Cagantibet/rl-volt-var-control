"""The single reward/score function used for training and for scoring every strategy."""

from __future__ import annotations

import numpy as np

from .config import RewardConfig


def compute_reward(v: np.ndarray, p_kw: np.ndarray, q_kvar: np.ndarray, *,
                   fleet_pmpp_kw: float, fleet_q_kvar: float,
                   band: tuple[float, float], cfg: RewardConfig) -> float:
    """Score one network state.

    v: per-unit voltage of every node. p_kw, q_kvar: measured PV output per inverter.
    """
    v = np.asarray(v, dtype=float)
    lo, hi = band
    under = v < lo
    over = v > hi
    ideal = (v >= cfg.ideal_band[0]) & (v <= cfg.ideal_band[1])
    acceptable = ~under & ~over & ~ideal
    n_viol = int(under.sum() + over.sum())

    reward = cfg.ideal_bonus * ideal.sum() + cfg.acceptable_bonus * acceptable.sum()
    reward -= cfg.undervoltage_weight * (lo - v[under]).sum()
    reward -= cfg.overvoltage_weight * (v[over] - hi).sum()
    reward -= cfg.violation_penalty * n_viol

    utilization = float(np.sum(np.abs(p_kw))) / fleet_pmpp_kw
    w = cfg.utilization_weight_ok if n_viol == 0 else cfg.utilization_weight_bad
    reward += w * utilization

    std = float(np.std(v))
    if std < 0.01:
        reward += cfg.balance_bonus[0]
    elif std < 0.02:
        reward += cfg.balance_bonus[1]

    reward -= cfg.reactive_weight * float(np.sum(np.abs(q_kvar))) / fleet_q_kvar
    return float(reward)
