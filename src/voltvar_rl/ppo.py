"""Proximal Policy Optimization (clipped surrogate) with GAE, for a Gaussian policy."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from .config import PPOConfig


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


class ActorCritic(nn.Module):
    """Shared MLP trunk, tanh-bounded Gaussian mean head, scalar value head."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig):
        super().__init__()
        h = cfg.hidden_dim
        self.trunk = nn.Sequential(nn.Linear(obs_dim, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.actor = nn.Sequential(nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, act_dim), nn.Tanh())
        self.critic = nn.Sequential(nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, 1))
        self.log_std = nn.Parameter(torch.full((act_dim,), math.log(cfg.init_std)))
        self.log_std_bounds = (math.log(cfg.min_std), math.log(cfg.max_std))

    def forward(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        z = self.trunk(obs)
        mean = self.actor(z)
        std = self.log_std.clamp(*self.log_std_bounds).exp().expand_as(mean)
        return Normal(mean, std), self.critic(z).squeeze(-1)


@dataclass
class Rollout:
    obs: list
    actions: list
    logprobs: list
    values: list
    rewards: list
    ends: list         # True on the last step of an episode (terminal or truncated)
    bootstrap: list    # V(s_{t+1}) on truncated steps, else 0

    @classmethod
    def empty(cls) -> "Rollout":
        return cls([], [], [], [], [], [], [])

    def __len__(self) -> int:
        return len(self.rewards)


def compute_gae(rewards: np.ndarray, values: np.ndarray, ends: np.ndarray, bootstrap: np.ndarray,
                gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation. Advantages never cross episode boundaries.

    On a truncated last step, the value of the next state (``bootstrap``) is used.
    """
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float64)
    last = 0.0
    for t in reversed(range(n)):
        if ends[t]:
            next_value, last = bootstrap[t], 0.0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


class PPO:
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig):
        self.cfg = cfg
        self.net = ActorCritic(obs_dim, act_dim, cfg)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        """Return (action, log-prob, value) for one observation."""
        dist, value = self.net(torch.as_tensor(obs, dtype=torch.float32))
        action = dist.mean if deterministic else dist.sample()
        return action.numpy(), float(dist.log_prob(action).sum()), float(value)

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> float:
        return float(self.net(torch.as_tensor(obs, dtype=torch.float32))[1])

    def update(self, ro: Rollout, generator: torch.Generator) -> dict[str, float]:
        c = self.cfg
        rewards = np.asarray(ro.rewards, dtype=np.float64) * c.reward_scale
        values = np.asarray(ro.values, dtype=np.float64)
        adv, ret = compute_gae(rewards, values, np.asarray(ro.ends), np.asarray(ro.bootstrap),
                               c.gamma, c.gae_lambda)

        obs = torch.as_tensor(np.stack(ro.obs), dtype=torch.float32)       # (N, obs_dim)
        actions = torch.as_tensor(np.stack(ro.actions), dtype=torch.float32)  # (N, act_dim)
        old_logp = torch.as_tensor(ro.logprobs, dtype=torch.float32)        # (N,)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)                   # (N,)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)                   # (N,)

        n = len(ro)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        n_batches = 0
        for _ in range(c.epochs):
            perm = torch.randperm(n, generator=generator)
            for start in range(0, n, c.minibatch_size):
                idx = perm[start:start + c.minibatch_size]
                dist, v = self.net(obs[idx])
                logp = dist.log_prob(actions[idx]).sum(-1)    # (B,)
                entropy = dist.entropy().sum(-1).mean()
                a = adv_t[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = torch.exp(logp - old_logp[idx])        # (B,)
                surr = torch.min(ratio * a, ratio.clamp(1 - c.clip_eps, 1 + c.clip_eps) * a)
                policy_loss = -surr.mean()
                value_loss = ((v - ret_t[idx]) ** 2).mean()
                loss = policy_loss + c.value_coef * value_loss - c.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.opt.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                stats["approx_kl"] += (old_logp[idx] - logp).mean().item()
                n_batches += 1
        return {k: v / n_batches for k, v in stats.items()}

    def save(self, path) -> None:
        torch.save({"model": self.net.state_dict(), "optimizer": self.opt.state_dict()}, path)

    def load(self, path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.net.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.opt.load_state_dict(ckpt["optimizer"])
