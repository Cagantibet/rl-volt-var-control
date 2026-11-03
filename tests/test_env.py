import numpy as np
import opendssdirect as dss

from voltvar_rl import curves as C
from voltvar_rl.evaluate import run_episode
from voltvar_rl.policies import baselines


def test_clock_does_not_drift(env):
    """In OpenDSS Daily mode each Solve advances the clock, so solve_at must reset the hour."""
    env.reset(2001, 12)
    p0 = env.feeder.pv_powers()[0].sum()
    for _ in range(5):
        env.step_curves(C.vv_disabled(env.n_pv), C.vw_disabled(env.n_pv))
        assert env.feeder.hour == 12
    np.testing.assert_allclose(env.feeder.pv_powers()[0].sum(), p0, rtol=1e-6)


def test_solve_at_uses_the_requested_hour(env):
    env.reset(2001, 12)
    p12 = env.feeder.pv_powers()[0].sum()
    env.feeder.solve_at(12)
    env.feeder.solve_at(12)
    assert np.isclose(env.feeder.pv_powers()[0].sum(), p12)
    assert np.isclose(p12, env.feeder.available_power(12).sum(), rtol=1e-3)


def test_pv_output_is_not_double_counted_and_curtailment_works(env):
    env.reset(2001, 12)
    avail = env.feeder.available_power(12)
    env.feeder.set_pv_setpoints(0.5 * avail, np.zeros(env.n_pv))
    env.feeder.solve_at(12)
    p, _ = env.feeder.pv_powers()
    np.testing.assert_allclose(p, 0.5 * avail, rtol=1e-3)


def test_no_pv_output_at_night(env):
    env.reset(2001, 2)
    assert env.feeder.pv_powers()[0].sum() < 1e-6


def test_reset_is_reproducible(env):
    a = env.reset(2002, 13)
    env.step(np.zeros(env.act_dim))
    b = env.reset(2002, 13)
    np.testing.assert_array_equal(a, b)


def test_episode_has_fixed_length(env, cfg):
    env.reset(2001, 12)
    steps, done = 0, False
    while not done:
        _, _, done, _ = env.step(np.zeros(env.act_dim))
        steps += 1
    assert steps == cfg.env.steps_per_episode


def test_volt_var_reduces_overvoltage(env):
    by_name = {p.name: p for p in baselines()}
    none = run_episode(env, by_name["No control"], 2001, 12)
    vv = run_episode(env, by_name["Volt-VAR"], 2001, 12)
    assert none.v_max > 1.05
    assert vv.v_max < none.v_max and vv.q_abs_kvar > 0


def test_node_distances_are_available(env):
    """Without an EnergyMeter OpenDSS reports every bus distance as 0."""
    d = env.feeder.node_distances
    assert d.max() > 1.5 and (d > 0).mean() > 0.9
