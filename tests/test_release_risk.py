"""Tests for src/release_risk.py (no download: small synthetic images)."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.attack_sum import match_and_order
from src import metrics, release_risk as rr


def _images(n, seed=0, c=1, hw=16):
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(n, c, hw, hw, generator=g)
    return torch.nn.functional.avg_pool2d(base, 3, 1, 1)  # some structure, not pure noise


def test_ssim_similarity_matches_metrics_ssim():
    pool = _images(6, seed=1, c=3)
    q = _images(2, seed=2, c=3)
    scores = rr.SSIMSimilarity()(q, pool)
    for i in range(2):
        for j in range(6):
            assert scores[i, j].item() == pytest.approx(metrics.ssim(q[i], pool[j]), abs=1e-5)
    assert rr.ssim_many(q[0], pool)[3].item() == pytest.approx(metrics.ssim(q[0], pool[3]), abs=1e-5)


def test_match_agrees_with_study_matching():
    targets = _images(5, seed=3)
    recons = (targets[[2, 0, 4, 1, 3]] + 0.1 * _images(5, seed=4)).clamp(0, 1)
    ssims, ordered, order = rr.match(recons, targets)
    records, study_ordered = match_and_order(recons, list(targets))
    assert order == [1, 3, 0, 4, 2]
    assert torch.allclose(ordered, study_ordered)
    assert ssims == pytest.approx([r["ssim"] for r in records], abs=1e-5)


def test_mean_ssim_and_any_pass():
    targets = _images(3, seed=5)
    perfect = rr.score_release(targets, targets[[2, 1, 0]], [0, 1, 2])
    assert perfect["mean_ssim"] == pytest.approx(1.0, abs=1e-5)
    assert perfect["any_pass"] == 1
    blank = rr.score_release(targets, torch.zeros_like(targets), [0, 1, 2])
    assert blank["any_pass"] == 0
    assert rr.any_passes([0.1, 0.9]) == 1 and rr.any_passes([0.1, 0.89]) == 0


def test_matched_flag_skips_matching():
    targets = _images(2, seed=6)
    swapped = targets[[1, 0]]
    assert rr.score_release(targets, swapped, [0, 0], matched=True)["mean_ssim"] < 0.9
    assert rr.score_release(targets, swapped, [0, 0])["mean_ssim"] == pytest.approx(1.0, abs=1e-5)


def _gallery(similarity=None):
    imgs = _images(12, seed=7)
    labels = np.array([0] * 6 + [1] * 6)
    ids = np.arange(100, 112)
    return rr.Gallery(imgs, labels, ids, similarity), imgs, labels, ids


def test_retrieval_hits_when_recon_is_a_member():
    gallery, imgs, labels, ids = _gallery()
    assert gallery.sizes() == {0: 6, 1: 6}
    targets, release_ids = imgs[[1, 8]], ids[[1, 8]]
    out = rr.score_release(targets, targets.clone(), labels[[1, 8]], release_ids, gallery)
    assert out["hit"] == 1 and sorted(out["picks"]) == [101, 108]


def test_retrieval_is_blind_to_membership():
    gallery, imgs, labels, ids = _gallery()
    recon = imgs[[3]]
    hit_in, picks_in = rr.retrieval_hit(recon, [0], [103], gallery)
    hit_out, picks_out = rr.retrieval_hit(recon, [0], [104], gallery)
    assert picks_in == picks_out == [103]
    assert hit_in == 1 and hit_out == 0


def test_retrieval_only_searches_the_given_class():
    gallery, imgs, labels, ids = _gallery()
    # An exact copy of a class-0 image, searched in class 1, can never pick it.
    hit, picks = rr.retrieval_hit(imgs[[2]], [1], [102], gallery)
    assert hit == 0 and picks[0] in set(ids[6:].tolist())


def test_recon_classes_follow_the_reconstruction_through_matching():
    gallery, imgs, labels, ids = _gallery()
    targets = imgs[[0, 7]]
    recons = imgs[[7, 0]]  # given in swapped order
    out = rr.score_release(targets, recons, [0, 1], ids[[0, 7]], gallery, recon_classes=[1, 0])
    assert out["hit"] == 1 and sorted(out["picks"]) == [100, 107]


def test_similarity_is_pluggable():
    gallery, imgs, labels, ids = _gallery(similarity=rr.neg_mse)
    hit, picks = rr.retrieval_hit(imgs[[9]] + 0.01, [1], [109], gallery)
    assert hit == 1 and picks == [109]


def test_label_only_guess_scores_like_a_reconstruction():
    gallery, imgs, labels, ids = _gallery()
    class_avg = torch.stack([imgs[labels == c].mean(0) for c in (0, 1)])
    targets = imgs[[0, 6]]
    out = rr.score_release(targets, class_avg, [0, 1], ids[[0, 6]], gallery, recon_classes=[0, 1])
    assert set(out) == {"mean_ssim", "ssims", "any_pass", "hit", "picks"}
    assert out["hit"] in (0, 1)
