"""Seeded 24-hour load and irradiance profiles.

A scenario is identified by ``(seed, day)``. The same pair always gives the same profiles,
whichever policy is being evaluated and however many times it is called.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ProfileConfig

HOURS = 24


@dataclass(frozen=True)
class DayProfiles:
    load: np.ndarray        # (24,) load multiplier
    irradiance: np.ndarray  # (n_pv, 24) irradiance in pu of 1 kW/m^2
    clearness: float


def clear_sky(sunrise: int, sunset: int) -> np.ndarray:
    """Clear-sky irradiance shape (0 at night, 1 at solar noon) for hours 0..23."""
    h = np.arange(HOURS, dtype=float)
    shape = np.sin(np.pi * (h - sunrise) / (sunset - sunrise))
    return np.clip(shape, 0.0, None)


def make_day(cfg: ProfileConfig, seed: int, day: int, n_pv: int) -> DayProfiles:
    rng = np.random.default_rng([seed, day])

    base = np.asarray(cfg.load_base, dtype=float)
    noise = rng.normal(0.0, np.asarray(cfg.load_noise, dtype=float))
    load = np.clip(base + noise, *cfg.load_clip)

    clearness = float(rng.uniform(*cfg.clearness_range))
    pv_noise = rng.normal(1.0, cfg.irradiance_noise, size=(n_pv, HOURS))
    irr = np.clip(clear_sky(cfg.sunrise, cfg.sunset)[None, :] * clearness * pv_noise, 0.0, 1.0)

    return DayProfiles(load=load, irradiance=irr, clearness=clearness)
