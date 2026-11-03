import numpy as np

from voltvar_rl import curves as C


def test_evaluate_interpolates_and_holds_ends():
    c = C.vv_ieee1547(1)
    q = lambda v: C.evaluate(c, np.array([v]))[0]
    assert q(0.80) == 0.44 and q(1.20) == -0.44
    assert q(1.00) == 0.0
    assert np.isclose(q(1.05), -0.22)


def test_action_mapping_gives_valid_curves(cfg):
    rng = np.random.default_rng(0)
    action = rng.normal(0, 3, size=15 * C.PARAMS_PER_PV)  # deliberately out of [-1, 1]
    vv, vw = C.action_to_curves(action, 15, cfg.curves, cfg.pv.q_max_pu)
    assert vv.shape == vw.shape == (15, 2, 5)
    assert np.all(np.diff(vv[:, 0], axis=1) >= 0) and np.all(np.diff(vw[:, 0], axis=1) >= 0)
    assert np.all(np.diff(vv[:, 1], axis=1) <= 0) and np.all(np.diff(vw[:, 1], axis=1) <= 0)
    lo, hi = cfg.curves.vv_voltage_range
    assert vv[:, 0].min() >= lo and vv[:, 0].max() <= hi
    assert np.abs(vv[:, 1]).max() <= cfg.pv.q_max_pu
    assert vw[:, 1].min() >= cfg.curves.vw_p_min and vw[:, 1].max() <= 1.0
