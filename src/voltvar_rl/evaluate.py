"""Evaluate strategies on identical, seeded scenarios."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .env import StepInfo, VoltVarEnv


def day_range(r) -> list[int]:
    return list(range(int(r[0]), int(r[1]) + 1))


def run_episode(env: VoltVarEnv, policy, day: int, hour: int) -> StepInfo:
    """Run one fixed-length episode and return the metrics of the final (converged) state."""
    obs = env.reset(day, hour)
    info = env.initial_info()
    done = False
    while not done:
        obs, _, done, info = env.step_curves(*policy.curves(env, obs))
    return info


def evaluate(env: VoltVarEnv, policies, days, hours) -> list[dict]:
    rows = []
    for policy in policies:
        for day in days:
            for hour in hours:
                info = run_episode(env, policy, day, hour)
                rows.append({"strategy": policy.name, "day": day, "hour": hour, **asdict(info),
                             "curtailment_pct": info.curtailment_pct})
    return rows


SUMMARY_FIELDS = ["reward", "n_violations", "v_max", "v_min", "p_kw", "curtailment_pct", "q_abs_kvar"]


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for name in dict.fromkeys(r["strategy"] for r in rows):
        sub = [r for r in rows if r["strategy"] == name]
        s = {"strategy": name, "scenarios": len(sub),
             "violation_free_pct": 100.0 * np.mean([r["n_violations"] == 0 for r in sub])}
        for f in SUMMARY_FIELDS:
            vals = np.array([r[f] for r in sub], dtype=float)
            s[f"{f}_mean"] = float(vals.mean())
            s[f"{f}_std"] = float(vals.std())
        out.append(s)
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def summary_markdown(summary: list[dict]) -> str:
    lines = [
        "| Strategy | Reward | Violating nodes | Violation-free scenarios | Max V (pu) | Min V (pu) | PV output (kW) | Curtailment | Σ abs Q (kvar) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        lines.append(
            f"| {s['strategy']} | {s['reward_mean']:.1f} ± {s['reward_std']:.1f} "
            f"| {s['n_violations_mean']:.1f} ± {s['n_violations_std']:.1f} "
            f"| {s['violation_free_pct']:.0f} % "
            f"| {s['v_max_mean']:.4f} | {s['v_min_mean']:.4f} "
            f"| {s['p_kw_mean']:.0f} | {s['curtailment_pct_mean']:.1f} % | {s['q_abs_kvar_mean']:.0f} |"
        )
    return "\n".join(lines) + "\n"


SEED_FIELDS = ["reward", "n_violations", "violation_free_pct", "v_max", "p_kw", "curtailment_pct",
               "q_abs_kvar"]


def seed_aggregate(summary: list[dict], prefix: str = "PPO seed") -> dict | None:
    """Mean and std *across training seeds* of each seed's scenario-mean metrics."""
    seeds = [s for s in summary if s["strategy"].startswith(prefix)]
    if len(seeds) < 2:
        return None
    out = {"strategy": f"PPO ({len(seeds)} seeds)", "seeds": len(seeds)}
    for f in SEED_FIELDS:
        key = f if f == "violation_free_pct" else f"{f}_mean"
        vals = np.array([s[key] for s in seeds], dtype=float)
        out[f"{f}_mean"] = float(vals.mean())
        out[f"{f}_std"] = float(vals.std(ddof=1))
    return out


def seed_markdown(summary: list[dict], agg: dict, reference: str = "VV+VW tight") -> str:
    ref = next(s for s in summary if s["strategy"] == reference)
    lines = [
        "| Strategy | Reward | Violating nodes | Violation-free scenarios | Max V (pu) | PV output (kW) | Curtailment | Σ abs Q (kvar) |",
        "|---|---|---|---|---|---|---|---|",
        f"| {reference} (deterministic) | {ref['reward_mean']:.1f} | {ref['n_violations_mean']:.1f} "
        f"| {ref['violation_free_pct']:.0f} % | {ref['v_max_mean']:.4f} | {ref['p_kw_mean']:.0f} "
        f"| {ref['curtailment_pct_mean']:.1f} % | {ref['q_abs_kvar_mean']:.0f} |",
    ]
    for s in summary:
        if s["strategy"].startswith("PPO seed"):
            lines.append(
                f"| {s['strategy']} | {s['reward_mean']:.1f} | {s['n_violations_mean']:.1f} "
                f"| {s['violation_free_pct']:.0f} % | {s['v_max_mean']:.4f} | {s['p_kw_mean']:.0f} "
                f"| {s['curtailment_pct_mean']:.1f} % | {s['q_abs_kvar_mean']:.0f} |")
    a = agg
    lines.append(
        f"| **{a['strategy']}, mean ± std** | **{a['reward_mean']:.1f} ± {a['reward_std']:.1f}** "
        f"| **{a['n_violations_mean']:.1f} ± {a['n_violations_std']:.1f}** "
        f"| **{a['violation_free_pct_mean']:.0f} ± {a['violation_free_pct_std']:.0f} %** "
        f"| **{a['v_max_mean']:.4f} ± {a['v_max_std']:.4f}** "
        f"| **{a['p_kw_mean']:.0f} ± {a['p_kw_std']:.0f}** "
        f"| **{a['curtailment_pct_mean']:.1f} ± {a['curtailment_pct_std']:.1f} %** "
        f"| **{a['q_abs_kvar_mean']:.0f} ± {a['q_abs_kvar_std']:.0f}** |")
    return "\n".join(lines) + "\n"
