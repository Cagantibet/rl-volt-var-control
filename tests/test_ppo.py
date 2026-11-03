import numpy as np
import torch

from voltvar_rl.config import PPOConfig
from voltvar_rl.ppo import PPO, Rollout, compute_gae, set_global_seed


def test_gae_does_not_cross_episode_boundaries():
    r = np.array([1.0, 1.0, 5.0, 5.0])
    v = np.zeros(4)
    ends = np.array([False, True, False, True])
    adv, ret = compute_gae(r, v, ends, np.zeros(4), gamma=1.0, lam=1.0)
    np.testing.assert_allclose(adv, [2.0, 1.0, 10.0, 5.0])
    np.testing.assert_allclose(ret, adv)


def test_gae_bootstraps_on_truncation():
    adv, _ = compute_gae(np.array([0.0]), np.array([0.0]), np.array([True]), np.array([3.0]),
                         gamma=0.5, lam=1.0)
    assert np.isclose(adv[0], 1.5)


def test_policy_shapes_have_no_hidden_batch_dim():
    """Log-probs and values must be (N,), or ratio * advantage silently broadcasts to (N, N)."""
    set_global_seed(0)
    cfg = PPOConfig(rollout_steps=8, minibatch_size=4, epochs=1)
    agent = PPO(obs_dim=6, act_dim=3, cfg=cfg)
    obs = np.zeros(6, dtype=np.float32)
    action, logp, value = agent.act(obs)
    assert action.shape == (3,) and isinstance(logp, float) and isinstance(value, float)
    dist, v = agent.net(torch.zeros(5, 6))
    assert dist.log_prob(torch.zeros(5, 3)).sum(-1).shape == (5,) and v.shape == (5,)


def test_ppo_learns_a_simple_bandit():
    set_global_seed(0)
    cfg = PPOConfig(rollout_steps=256, minibatch_size=64, epochs=5, lr=3e-3, reward_scale=1.0)
    agent = PPO(obs_dim=2, act_dim=1, cfg=cfg)
    gen = torch.Generator().manual_seed(0)
    obs = np.ones(2, dtype=np.float32)
    for _ in range(30):
        ro = Rollout.empty()
        for _ in range(cfg.rollout_steps):
            a, lp, v = agent.act(obs)
            ro.obs.append(obs); ro.actions.append(a); ro.logprobs.append(lp); ro.values.append(v)
            ro.rewards.append(-float((a[0] - 0.5) ** 2)); ro.ends.append(True); ro.bootstrap.append(0.0)
        agent.update(ro, gen)
    mean_action, _, _ = agent.act(obs, deterministic=True)
    assert abs(mean_action[0] - 0.5) < 0.1


def test_save_load_roundtrip(tmp_path):
    cfg = PPOConfig()
    a, b = PPO(4, 2, cfg), PPO(4, 2, cfg)
    a.save(tmp_path / "m.pt")
    b.load(tmp_path / "m.pt")
    obs = np.ones(4, dtype=np.float32)
    np.testing.assert_allclose(a.act(obs, True)[0], b.act(obs, True)[0])
