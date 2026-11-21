"""Tests for the per-release journal baselines (scripts/baselines.py --journal)."""

import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import baselines
from src import release_risk as rr


def _world(seed=0, n_img=40, hw=16):
    g = torch.Generator().manual_seed(seed)
    imgs = torch.nn.functional.avg_pool2d(torch.rand(n_img, 1, hw, hw, generator=g), 3, 1, 1)
    labels = np.array([i % 2 for i in range(n_img)])
    gallery = rr.Gallery(imgs, labels, np.arange(n_img))
    class_avg = torch.stack([imgs[labels == c].mean(0) for c in (0, 1)])
    return imgs, labels, gallery, class_avg


def test_release_baseline_follows_plan_section_10():
    imgs, labels, gallery, class_avg = _world()
    release = [0, 3]
    prior = list(range(20, 40))
    prior_by_class = {c: np.array([i for i in prior if labels[i] == c]) for c in (0, 1)}
    rec = baselines.release_baseline(2, 7, imgs[release], labels[release], release, class_avg, prior_by_class,
                                     lambda i: imgs[i], gallery, imgs.mean(0))
    # Seed and draw order: default_rng(50000 + 1000 N + r), draw 0..4, then image order.
    rng = np.random.default_rng(50000 + 2000 + 7)
    want = [[int(rng.choice(prior_by_class[labels[i]])) for i in release] for _ in range(5)]
    assert [d["ids"] for d in rec["same_class_draws"]] == want
    assert all(i in prior for d in rec["same_class_draws"] for i in d["ids"])
    assert all(labels[d["ids"][j]] == labels[release[j]] for d in rec["same_class_draws"] for j in range(2))
    assert rec["S"] == pytest.approx(np.mean([d["mean_ssim"] for d in rec["same_class_draws"]]))
    assert rec["b"] == max(rec["A"], rec["S"])
    share = np.mean([d["any_pass"] for d in rec["same_class_draws"]])
    assert rec["baseline_any_pass"] == max(rec["any_pass_class_average"], share)
    assert rec["baseline_hit"] == max(rec["hit_class_average"], rec["hit_same_share"])
    assert rec["hit_exact_chance"] == pytest.approx(rr.chance_hit(labels[release], gallery.sizes()))
    assert set(rec["context_mean_ssim"]) == {"all black", "average digit", "random noise"}


def test_a_guess_never_retrieves_itself():
    imgs, labels, gallery, class_avg = _world()
    # The prior holds one image per class, so every draw is the same pair of images.
    prior_by_class = {0: np.array([20]), 1: np.array([21])}
    rec = baselines.release_baseline(2, 0, imgs[[0, 1]], labels[[0, 1]], [0, 1], class_avg, prior_by_class,
                                     lambda i: imgs[i], gallery)
    for d in rec["same_class_draws"]:
        assert d["ids"] == [20, 21] and 20 not in d["picks"] and 21 not in d["picks"]
