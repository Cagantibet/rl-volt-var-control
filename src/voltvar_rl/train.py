"""PPO training loop."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import Config
from .env import VoltVarEnv
from .evaluate import day_range, evaluate
from .policies import PPOPolicy
from .ppo import PPO, Rollout, set_global_seed


def train(cfg: Config, out_dir: Path, log=print) -> Path:
    """Train PPO and return the path of the best checkpoint (selected on validation days)."""
    set_global_seed(cfg.seed)
    out_dir = Path(out_dir).resolve()  # OpenDSS's Compile changes the process working directory
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)

    env = VoltVarEnv(cfg)
    agent = PPO(env.obs_dim, env.act_dim, cfg.ppo)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    train_days = day_range(cfg.env.train_days)
    val_days = day_range(cfg.env.val_days)
    hours = list(cfg.env.hours)

    def validate() -> float:
        rows = evaluate(env, [PPOPolicy(agent)], val_days, hours)
        return float(np.mean([r["reward"] for r in rows]))

    best_path, final_path = out_dir / "ppo_best.pt", out_dir / "ppo_final.pt"
    history = [{"step": 0, "val_reward": validate(), "train_final_reward": float("nan")}]
    best = history[0]["val_reward"]
    agent.save(best_path)
    log(f"step 0 | val reward {best:.1f}")

    step, next_eval, t0 = 0, cfg.train.eval_every, time.time()
    episode_finals: list[float] = []
    while step < cfg.train.total_steps:
        ro = Rollout.empty()
        while len(ro) < cfg.ppo.rollout_steps:
            obs = env.reset(int(rng.choice(train_days)), int(rng.choice(hours)))
            done = False
            while not done:
                action, logp, value = agent.act(obs)
                next_obs, reward, done, _ = env.step(action)
                ro.obs.append(obs)
                ro.actions.append(action)
                ro.logprobs.append(logp)
                ro.values.append(value)
                ro.rewards.append(reward)
                ro.ends.append(done)
                ro.bootstrap.append(agent.value(next_obs) if done else 0.0)
                obs = next_obs
            episode_finals.append(reward)
        step += len(ro)
        stats = agent.update(ro, gen)

        if step >= next_eval or step >= cfg.train.total_steps:
            next_eval += cfg.train.eval_every
            val = validate()
            recent = float(np.mean(episode_finals[-50:]))
            history.append({"step": step, "val_reward": val, "train_final_reward": recent, **stats})
            marker = ""
            if val > best:
                best, marker = val, "  (new best)"
                agent.save(best_path)
            log(f"step {step} | train final reward {recent:.1f} | val reward {val:.1f} | "
                f"std {agent.net.log_std.detach().exp().mean().item():.3f} | {time.time() - t0:.0f}s{marker}")

    agent.save(final_path)
    keys = sorted({k for h in history for k in h}, key=lambda k: (k != "step", k))
    with open(out_dir / "training_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)
    with open(out_dir / "train_summary.json", "w") as f:
        json.dump({"best_val_reward": best, "total_steps": step,
                   "wall_time_s": round(time.time() - t0, 1)}, f, indent=2)
    return best_path
