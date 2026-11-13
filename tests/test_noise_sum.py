"""Tests for the clip + sum + noise mechanism of scripts/attack_noise_sum.py (no download)."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.attack_noise_sum import epsilon_for, noisy_mean_gradient


def _grads(k, scale=1.0):
    gen = torch.Generator().manual_seed(0)
    return [[torch.randn(3, 4, generator=gen) * scale, torch.randn(5, generator=gen) * scale] for _ in range(k)]


def test_no_noise_large_clip_is_plain_mean():
    per_example = _grads(4)
    out, frac = noisy_mean_gradient(per_example, clip=1e6, sigma=0.0)
    assert frac == 0.0
    for i, o in enumerate(out):
        assert torch.equal(o, sum(g[i] for g in per_example) / 4)


def test_clipped_examples_have_norm_at_most_clip():
    per_example = _grads(3, scale=10.0)
    clip = 1.0
    # With one example and no noise, the output is exactly that clipped example.
    for grads in per_example:
        out, frac = noisy_mean_gradient([grads], clip=clip, sigma=0.0)
        assert frac == 1.0
        assert torch.sqrt(sum((o ** 2).sum() for o in out)) <= clip + 1e-5


def test_noise_is_added_once_with_std_sigma_clip_over_k():
    k, clip, sigma = 4, 2.0, 0.5
    per_example = [[torch.zeros(100_000)] for _ in range(k)]
    out, _ = noisy_mean_gradient(per_example, clip=clip, sigma=sigma, seed=1)
    assert abs(out[0].std().item() - sigma * clip / k) < 0.01


def test_epsilon_falls_as_sigma_rises():
    assert epsilon_for(0.0) == float("inf")
    assert epsilon_for(4.0) < epsilon_for(1.0) < epsilon_for(0.1)


def test_matched_sigma_equals_sum_relative_noise():
    # Each example's relative noise sigma'_i*C/||g_i|| must equal the sum's sigma*C/||sum g||.
    from scripts.attack_matched_snr import global_norm, matched_sigmas
    torch.manual_seed(0)
    per_example = [[torch.randn(5, 3) * s, torch.randn(4) * s] for s in (1.0, 2.0, 0.5)]
    clip, sigma = 100.0, 1e-3
    eff = matched_sigmas(per_example, clip, sigma)
    sum_norm = global_norm([sum(p) for p in zip(*per_example)])
    for grads, s in zip(per_example, eff):
        assert abs(s * clip / global_norm(grads) - sigma * clip / sum_norm) < 1e-9
    # A single example is its own sum, so sigma is unchanged.
    assert abs(matched_sigmas(per_example[:1], clip, sigma)[0] - sigma) < 1e-12
