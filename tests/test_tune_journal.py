"""Tests for the journal TV tuning rules (scripts/tune_journal.py)."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import tune_journal as tj


def test_grid_and_tuning_pairs_follow_plan_section_8_2():
    assert tj.TVS == [1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    pairs = tj.tuning_pairs()
    assert len(pairs) == 12 and pairs[11]["image_indices"] == [22, 23]
    assert [p["checkpoint"] for p in pairs] == [0] * 4 + [1] * 4 + [2] * 4


def test_restart_seeds_follow_plan_section_8_2():
    class A:
        setting, restarts, local_lr, iters, cosine_iters, unrolled_iters = "mnist-fresh", 2, 0.3, 1, 1, 1
        out, git_commit = "/tmp/x", None
    t = tj.make_task(A, "T-C", tj.tuning_pairs()[3], (2, 1), "unrolled", 1e-3, 5)
    assert t["restart_seeds"] == [90030, 90031] and t["checkpoint"] == 0 and t["image_indices"] == [6, 7]


def test_pick_tv_takes_the_highest_mean_and_breaks_ties_to_the_smaller_tv():
    assert tj.pick_tv({1e-4: 0.5, 1e-3: 0.7, 1e-2: 0.6}) == 1e-3
    assert tj.pick_tv({1e-4: 0.7, 1e-3: 0.7, 1e-2: 0.6}) == 1e-4


def test_objective_rule():
    assert tj.main_objective(0.62, 0.60) == "basic"  # higher by exactly 0.02
    assert tj.main_objective(0.619, 0.60) == "cosine"
    assert tj.main_objective(0.3, 0.6) == "cosine"


class _Args:
    setting, restarts, local_lr, iters, cosine_iters, unrolled_iters = "mnist-fresh", 2, 0.3, 1, 1, 1
    out, git_commit, workers = "/tmp/x", None, 1


@pytest.mark.parametrize("winner,extra,edge", [(1e-5, [1e-6], None), (1e-1, [1.0], None), (1e-3, [], None),
                                               (1e-6, [1e-6], "low"), (1.0, [1.0], "high")])
def test_edge_rule_extends_once_per_side(monkeypatch, winner, extra, edge):
    calls = []

    def fake_run(tasks, workers, label):
        calls.append(sorted({t["tv"] for t in tasks}))
        # SSIM falls with distance from `winner` (in decades).
        import math
        return [{"tv": t["tv"], "split": t["split"], "seconds": 1.0,
                 "ssim": [1.0 - 0.01 * abs(math.log10(t["tv"]) - math.log10(winner))]} for t in tasks]

    monkeypatch.setattr(tj, "run", fake_run)
    out = tj.tune_cell(_Args, "T-B")
    assert out["chosen"] == winner and out["extended_with"] == extra and out["at_edge"] == edge
    assert len(calls) == 1 + len(extra)
