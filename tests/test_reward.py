import numpy as np

from voltvar_rl.reward import compute_reward


def _r(cfg, v, p=0.0, q=0.0):
    return compute_reward(np.asarray(v), np.array([p]), np.array([q]), fleet_pmpp_kw=100.0,
                          fleet_q_kvar=100.0, band=(0.95, 1.05), cfg=cfg.reward)


def test_ideal_voltages_score_best(cfg):
    assert _r(cfg, [1.0] * 10) > _r(cfg, [1.03] * 10) > _r(cfg, [1.06] * 10)


def test_overvoltage_penalty_matches_definition(cfg):
    w = cfg.reward
    # one node at 1.06: penalty 150*0.01 + 5; std of a single value is 0 -> balance bonus
    expected = -(w.overvoltage_weight * 0.01) - w.violation_penalty + w.balance_bonus[0]
    assert np.isclose(_r(cfg, [1.06]), expected)


def test_power_rewarded_more_when_no_violations(cfg):
    w = cfg.reward
    assert np.isclose(_r(cfg, [1.0], p=50) - _r(cfg, [1.0]), w.utilization_weight_ok * 0.5)
    assert np.isclose(_r(cfg, [1.06], p=50) - _r(cfg, [1.06]), w.utilization_weight_bad * 0.5)


def test_reactive_power_is_penalised(cfg):
    assert _r(cfg, [1.0], q=-40) < _r(cfg, [1.0])
