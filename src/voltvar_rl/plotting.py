"""Figures for training, evaluation and simulations (static PNGs for the README)."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated categorical slots (first three pass all-pairs CVD checks), neutral inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
NEUTRAL = "#a9a8a3"
INK = "#0b0b0b"
INK_2 = "#52514e"
LIMIT = "#e34948"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2.0,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def _band(ax, band, axis="y"):
    for b in band:
        (ax.axhline if axis == "y" else ax.axvline)(b, color=LIMIT, ls="--", lw=1.0)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_training(logs: dict[str, Path], path: Path, baseline_val: dict[str, float] | None = None):
    """logs: run label -> training_log.csv. Up to three runs (one colour each)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for color, (label, log_csv) in zip(SERIES, logs.items()):
        with open(log_csv) as f:
            rows = list(csv.DictReader(f))
        step = np.array([float(r["step"]) for r in rows]) / 1000
        val = np.array([float(r["val_reward"]) for r in rows])
        name = "PPO" if label == "PPO" else f"PPO {label}"
        ax.plot(step, val, color=color, label=f"{name} (validation days)")
    for i, (name, v) in enumerate((baseline_val or {}).items()):
        ax.axhline(v, color=[INK_2, NEUTRAL][i % 2], ls=":", lw=1.4, label=f"{name} (validation days)")
    ax.set_xlabel("Training steps (thousands)")
    ax.set_ylabel("Mean final-step reward")
    ax.set_title("PPO training progress")
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_comparison(summary: list[dict], path: Path):
    """Four small panels: reward, violating nodes, curtailment, max voltage."""
    names = [s["strategy"] for s in summary]
    colors = [SERIES[0] if n.startswith("PPO") else NEUTRAL for n in names]
    panels = [("reward", "Reward (higher is better)", "{:.0f}"),
              ("n_violations", "Nodes outside 0.95-1.05 pu", "{:.1f}"),
              ("curtailment_pct", "PV curtailment (%)", "{:.1f}"),
              ("v_max", "Max node voltage (pu)", "{:.3f}")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    y = np.arange(len(names))
    for ax, (key, title, fmt) in zip(axes.flat, panels):
        mean = np.array([s[f"{key}_mean"] for s in summary])
        std = np.array([s[f"{key}_std"] for s in summary])
        ax.barh(y, mean, xerr=std, color=colors, height=0.6,
                error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        for yi, m in zip(y, mean):
            ax.annotate(fmt.format(m), (m, yi), xytext=(4, 6), textcoords="offset points",
                        fontsize=8, color=INK_2)
        if key == "v_max":
            ax.set_xlim(min(1.0, mean.min() - 0.01), max(mean + std) + 0.005)
            ax.axvline(1.05, color=LIMIT, ls="--", lw=1.0)
    fig.suptitle("Held-out test scenarios (mean ± std)", fontweight="bold")
    _save(fig, path)


def plot_voltage_profiles(profiles: dict[str, tuple[np.ndarray, np.ndarray]], band, path: Path,
                          title: str):
    """profiles: name -> (distance_km, voltage) for every node; one panel per strategy."""
    n = len(profiles)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), sharey=True)
    for ax, (name, (dist, v)) in zip(np.atleast_1d(axes), profiles.items()):
        ax.scatter(dist, v, s=10, color=SERIES[0], edgecolors="none", alpha=0.8)
        _band(ax, band)
        ax.set_title(name)
        ax.set_xlabel("Distance from substation (km)")
    np.atleast_1d(axes)[0].set_ylabel("Node voltage (pu)")
    fig.suptitle(title, fontweight="bold")
    _save(fig, path)


def plot_learned_curves(vv: np.ndarray, vw: np.ndarray, ref_vv: np.ndarray, ref_vw: np.ndarray,
                        labels: list[str], path: Path, title: str):
    v = np.linspace(0.90, 1.10, 300)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for i in range(len(vv)):
        a1.plot(v, np.interp(v, vv[i, 0], vv[i, 1]), color=SERIES[0], alpha=0.35, lw=1.2,
                label="PPO, one line per inverter" if i == 0 else None)
        a2.plot(v, 100 * np.interp(v, vw[i, 0], vw[i, 1]), color=SERIES[0], alpha=0.35, lw=1.2,
                label="PPO, one line per inverter" if i == 0 else None)
    a1.plot(v, np.interp(v, ref_vv[0], ref_vv[1]), color=INK, ls="--", lw=1.4, label=labels[0])
    a2.plot(v, 100 * np.interp(v, ref_vw[0], ref_vw[1]), color=INK, ls="--", lw=1.4, label=labels[1])
    a1.set(xlabel="Local voltage (pu)", ylabel="Reactive power (pu of kVA, + = inject)", title="Volt/VAR")
    a2.set(xlabel="Local voltage (pu)", ylabel="Active-power ceiling (% of available)", title="Volt/Watt")
    for ax in (a1, a2):
        _band(ax, (0.95, 1.05), axis="x")
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(title, fontweight="bold")
    _save(fig, path)


def plot_daily(daily: dict[str, list[dict]], band, path: Path, title: str):
    """daily: strategy -> 24 hourly StepInfo dicts. At most three strategies."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    hours = np.arange(24)
    others = iter(SERIES[1:])
    for name, rows in daily.items():
        color = SERIES[0] if name.startswith("PPO") else next(others)
        vmax = [r["v_max"] for r in rows]
        a1.plot(hours, vmax, color=color, label=name, marker="o", ms=4)
        a2.plot(hours, [r["p_kw"] for r in rows], color=color, label=name, marker="o", ms=4)
    _band(a1, band[1:])
    a1.set(xlabel="Hour of day", ylabel="Max node voltage (pu)", title="Worst-case voltage")
    a2.set(xlabel="Hour of day", ylabel="Total PV output (kW)", title="PV active power delivered")
    a1.legend(fontsize=8)
    fig.suptitle(title, fontweight="bold")
    _save(fig, path)
