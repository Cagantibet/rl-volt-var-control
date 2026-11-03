"""Experiment configuration: a YAML file loaded into typed dataclasses."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@dataclass
class FeederConfig:
    master: str
    regulator_tap: int = 4
    voltage_band: tuple[float, float] = (0.95, 1.05)


@dataclass
class PVConfig:
    kva: float = 200.0
    pmpp: float = 180.0
    kv: float = 2.4
    cutin_pct: float = 1.0
    q_max_pu: float = 0.44
    sites: dict[str, list[int]] = field(default_factory=dict)

    def site_list(self) -> list[tuple[str, int, int]]:
        """(name, bus, phase) for every PV system, in a fixed order."""
        out = []
        for phase_idx, phase in enumerate("abc", start=1):
            for bus in self.sites.get(phase, []):
                out.append((f"pv_{phase}_{bus}", bus, phase_idx))
        return out


@dataclass
class ProfileConfig:
    load_base: list[float]
    load_noise: list[float]
    load_clip: tuple[float, float] = (0.15, 1.15)
    sunrise: int = 6
    sunset: int = 18
    clearness_range: tuple[float, float] = (0.7, 1.0)
    irradiance_noise: float = 0.05


@dataclass
class CurveConfig:
    vv_voltage_range: tuple[float, float] = (0.90, 1.10)
    vw_voltage_range: tuple[float, float] = (1.00, 1.10)
    vw_p_min: float = 0.0


@dataclass
class RewardConfig:
    ideal_band: tuple[float, float] = (0.98, 1.02)
    ideal_bonus: float = 2.0
    acceptable_bonus: float = 0.5
    undervoltage_weight: float = 100.0
    overvoltage_weight: float = 150.0
    violation_penalty: float = 5.0
    utilization_weight_ok: float = 100.0
    utilization_weight_bad: float = 20.0
    balance_bonus: tuple[float, float] = (3.0, 1.5)
    reactive_weight: float = 2.5


@dataclass
class EnvConfig:
    steps_per_episode: int = 10
    damping: float = 0.5
    hours: list[int] = field(default_factory=lambda: [12])
    train_days: tuple[int, int] = (1, 500)
    val_days: tuple[int, int] = (1001, 1005)
    test_days: tuple[int, int] = (2001, 2020)


@dataclass
class PPOConfig:
    hidden_dim: int = 128
    init_std: float = 0.2
    min_std: float = 0.01
    max_std: float = 0.5
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 10
    minibatch_size: int = 256
    rollout_steps: int = 2000
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    reward_scale: float = 0.01


@dataclass
class TrainConfig:
    total_steps: int = 200_000
    eval_every: int = 10_000
    output_dir: str = "runs/default"


@dataclass
class Config:
    seed: int
    scenario_seed: int
    feeder: FeederConfig
    pv: PVConfig
    profiles: ProfileConfig
    curves: CurveConfig
    reward: RewardConfig
    env: EnvConfig
    ppo: PPOConfig
    train: TrainConfig

    @property
    def master_path(self) -> Path:
        return resolve_path(self.feeder.master)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


_SECTIONS = {
    "feeder": FeederConfig,
    "pv": PVConfig,
    "profiles": ProfileConfig,
    "curves": CurveConfig,
    "reward": RewardConfig,
    "env": EnvConfig,
    "ppo": PPOConfig,
    "train": TrainConfig,
}


def _build(cls, data: dict[str, Any]):
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            v = data[f.name]
            kwargs[f.name] = tuple(v) if isinstance(v, list) and "tuple" in str(f.type) else v
    return cls(**kwargs)


def config_from_dict(data: dict[str, Any]) -> Config:
    missing = set(_SECTIONS) - set(data)
    if missing:
        raise ValueError(f"Config is missing sections: {sorted(missing)}")
    return Config(seed=int(data.get("seed", 0)),
                  scenario_seed=int(data.get("scenario_seed", 0)),
                  **{name: _build(cls, data[name]) for name, cls in _SECTIONS.items()})


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a YAML config. ``overrides`` are ``section.key=value`` strings (value parsed as YAML)."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        data = yaml.safe_load(f)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return config_from_dict(data)
