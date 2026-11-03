"""Command-line interface: ``python -m voltvar_rl <command> [options]``."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import curves as C
from .config import DEFAULT_CONFIG, Config, load_config
from .env import VoltVarEnv
from .evaluate import (day_range, evaluate, run_episode, seed_aggregate, seed_markdown,
                       summarize, summary_markdown, write_csv)
from .policies import PPOPolicy, baselines
from .ppo import PPO, set_global_seed


def _load_agent(env: VoltVarEnv, cfg: Config, model: Path) -> PPO:
    agent = PPO(env.obs_dim, env.act_dim, cfg.ppo)
    agent.load(model)
    return agent


def _models(args) -> list[Path]:
    return [Path(m).resolve() for m in (args.model or [])]


def _ppo_name(path: Path, n_models: int) -> str:
    """'PPO' for a single model, 'PPO seed k' when several are compared (from runs/seed<k>)."""
    if n_models == 1:
        return "PPO"
    stem = path.parent.name if path.parent.name.startswith("seed") else path.stem
    return f"PPO {stem.replace('seed', 'seed ')}"


def _policies(env, cfg, models: list[Path]):
    pols = baselines()
    for m in models:
        pols.append(PPOPolicy(_load_agent(env, cfg, m), name=_ppo_name(m, len(models))))
    return pols


# ------------------------------------------------------------------ commands
def cmd_train(cfg: Config, args) -> None:
    from .plotting import plot_training
    from .train import train

    out = Path(args.out or cfg.train.output_dir).resolve()
    best = train(cfg, out)
    ref = _val_reference(cfg)
    plot_training({"PPO": out / "training_log.csv"}, out / "training_curve.png", ref)
    print(f"Best checkpoint: {best}")


def _val_reference(cfg: Config) -> dict[str, float]:
    env = VoltVarEnv(cfg)
    return {p.name: float(np.mean([r["reward"] for r in evaluate(env, [p], day_range(cfg.env.val_days),
                                                                   cfg.env.hours)]))
            for p in baselines() if p.name in ("VV+VW", "VV+VW tight")}


def cmd_plot_training(cfg: Config, args) -> None:
    from .plotting import plot_training

    runs = {Path(r).name.replace("seed", "seed "): Path(r).resolve() / "training_log.csv"
            for r in args.run}
    out = Path(args.out).resolve()
    plot_training(runs, out, _val_reference(cfg))
    print(f"Saved {out}")


def cmd_evaluate(cfg: Config, args) -> None:
    from .plotting import plot_comparison, plot_learned_curves

    out = Path(args.out).resolve()
    models = _models(args)
    env = VoltVarEnv(cfg)
    pols = _policies(env, cfg, models)
    days = day_range(cfg.env.test_days)
    rows = evaluate(env, pols, days, cfg.env.hours)
    summary = summarize(rows)

    write_csv(rows, out / "metrics_per_scenario.csv")
    write_csv(summary, out / "metrics_summary.csv")
    (out / "metrics_summary.md").write_text(summary_markdown(summary))
    plot_comparison(summary, out / "comparison.png")
    print(summary_markdown(summary))
    agg = seed_aggregate(summary)
    if agg is not None:
        write_csv([agg], out / "metrics_seeds.csv")
        (out / "metrics_seeds.md").write_text(seed_markdown(summary, agg))
        print(seed_markdown(summary, agg))

    if models:
        ppo = pols[len(baselines())]  # first model
        run_episode(env, ppo, days[0], 12)
        n = env.n_pv
        plot_learned_curves(env.vv, env.vw, C.vv_ieee1547(n)[0], C.vw_ieee1547(n)[0],
                            ["IEEE 1547 Cat. B default", "IEEE 1547 default"],
                            out / "learned_curves.png",
                            f"Curves chosen by {ppo.name} (test day {days[0]}, 12:00)")


def cmd_profile(cfg: Config, args) -> None:
    from .plotting import plot_voltage_profiles

    out = Path(args.out).resolve()
    models = _models(args)
    env = VoltVarEnv(cfg)
    dist = env.feeder.node_distances
    profiles = {}

    env.reset(args.day, args.hour)
    env.feeder.set_pv_enabled(False)
    env.feeder.solve_at(args.hour)
    profiles["No PV"] = (dist, env.feeder.node_voltages())
    env.feeder.set_pv_enabled(True)

    wanted = {"No control", "VV+VW tight"}
    for p in [p for p in _policies(env, cfg, models[:1]) if p.name in wanted or p.name.startswith("PPO")]:
        run_episode(env, p, args.day, args.hour)
        profiles[f"PV + {p.name}"] = (dist, env.v.copy())
    plot_voltage_profiles(profiles, cfg.feeder.voltage_band, out / "voltage_profiles.png",
                          f"Node voltages, day {args.day}, {args.hour}:00")
    print(f"Saved {out / 'voltage_profiles.png'}")


def cmd_simulate_day(cfg: Config, args) -> None:
    from .plotting import plot_daily

    out = Path(args.out).resolve()
    models = _models(args)
    env = VoltVarEnv(cfg)
    wanted = ["No control", "VV+VW tight", "PPO"]
    pols = [p for p in _policies(env, cfg, models[:1]) if p.name in wanted]
    daily = {p.name: [asdict(run_episode(env, p, args.day, h)) for h in range(24)] for p in pols}
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"daily_day{args.day}.json", "w") as f:
        json.dump(daily, f, indent=1)
    plot_daily(daily, cfg.feeder.voltage_band, out / "daily_simulation.png",
               f"24-hour simulation, test day {args.day} (policy trained on hours "
               f"{min(cfg.env.hours)}-{max(cfg.env.hours)})")
    for name, rows in daily.items():
        print(f"{name:12s} energy {sum(r['p_kw'] for r in rows):8.0f} kWh | "
              f"hours with violations {sum(r['n_violations'] > 0 for r in rows):2d}")


# ---------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config file")
    common.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set train.total_steps=50000")

    p = argparse.ArgumentParser(prog="voltvar_rl", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", parents=[common], help="train PPO")
    t.add_argument("--out", help="output directory (default: train.output_dir)")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", parents=[common], help="compare PPO with baselines on test days")
    e.add_argument("--model", action="append",
                   help="PPO checkpoint; repeat to compare several seeds (omit for baselines only)")
    e.add_argument("--out", default="results")
    e.set_defaults(func=cmd_evaluate)

    pr = sub.add_parser("profile", parents=[common], help="voltage profile snapshot for one day/hour")
    pr.add_argument("--model", action="append", help="PPO checkpoint (the first one is used)")
    pr.add_argument("--day", type=int, default=2001)
    pr.add_argument("--hour", type=int, default=12)
    pr.add_argument("--out", default="results")
    pr.set_defaults(func=cmd_profile)

    pt = sub.add_parser("plot-training", parents=[common], help="plot validation curves of runs")
    pt.add_argument("--run", action="append", required=True, help="training output directory")
    pt.add_argument("--out", default="results/training_curve.png")
    pt.set_defaults(func=cmd_plot_training)

    d = sub.add_parser("simulate-day", parents=[common], help="run every hour of one day")
    d.add_argument("--model", action="append", help="PPO checkpoint (the first one is used)")
    d.add_argument("--day", type=int, default=2001)
    d.add_argument("--out", default="results")
    d.set_defaults(func=cmd_simulate_day)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    set_global_seed(cfg.seed)
    args.func(cfg, args)


if __name__ == "__main__":
    main()
