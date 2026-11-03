import pytest

from voltvar_rl.config import load_config


def test_default_config_loads(cfg):
    assert cfg.master_path.exists()
    assert len(cfg.pv.site_list()) == 15
    assert cfg.feeder.voltage_band == (0.95, 1.05)


def test_overrides_are_applied():
    cfg = load_config(overrides=["train.total_steps=123", "env.hours=[9, 10]"])
    assert cfg.train.total_steps == 123
    assert cfg.env.hours == [9, 10]


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError):
        load_config(overrides=["ppo.not_a_key=1"])


def test_scenario_splits_are_disjoint(cfg):
    tr, va, te = cfg.env.train_days, cfg.env.val_days, cfg.env.test_days
    assert tr[1] < va[0] and va[1] < te[0]


def test_training_seed_does_not_change_scenarios():
    from voltvar_rl.profiles import make_day
    a, b = load_config(overrides=["seed=0"]), load_config(overrides=["seed=7"])
    assert a.scenario_seed == b.scenario_seed
    da = make_day(a.profiles, a.scenario_seed, 2001, 15)
    db = make_day(b.profiles, b.scenario_seed, 2001, 15)
    assert (da.load == db.load).all()
