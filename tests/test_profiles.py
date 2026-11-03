import numpy as np

from voltvar_rl.profiles import make_day


def test_same_seed_and_day_give_identical_profiles(cfg):
    a = make_day(cfg.profiles, 0, 7, 15)
    b = make_day(cfg.profiles, 0, 7, 15)
    np.testing.assert_array_equal(a.load, b.load)
    np.testing.assert_array_equal(a.irradiance, b.irradiance)


def test_different_days_differ(cfg):
    a = make_day(cfg.profiles, 0, 1, 15)
    b = make_day(cfg.profiles, 0, 2, 15)
    assert not np.allclose(a.load, b.load)


def test_profile_ranges(cfg):
    d = make_day(cfg.profiles, 0, 3, 15)
    assert d.load.shape == (24,) and d.irradiance.shape == (15, 24)
    assert d.load.min() >= cfg.profiles.load_clip[0] and d.load.max() <= cfg.profiles.load_clip[1]
    assert d.irradiance.min() >= 0 and d.irradiance.max() <= 1
    assert np.all(d.irradiance[:, :cfg.profiles.sunrise + 1] == 0)  # dark before sunrise
