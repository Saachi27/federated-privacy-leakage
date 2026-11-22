"""Tests for scripts/analyze_followup.py."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import analyze_followup as af  # noqa: E402


def _rec(release, labels, loss, perm=None, setting="mnist-fresh"):
    return {"setting": setting, "n": 2, "e": 5, "split_name": "2x1", "attacker": "unrolled",
            "release": release, "image_indices": [2 * release, 2 * release + 1], "checkpoint": release // 10,
            "checkpoint_hash": "h", "labels": labels, "final_loss": loss, "label_perm": perm}


def test_label_unknown_keeps_the_lower_objective_and_counts_distinct_labels():
    main = [_rec(0, [3, 5], 0.10), _rec(1, [3, 5], 0.30), _rec(2, [4, 4], 0.20), _rec(3, [1, 2], float("nan"))]
    swapped = [_rec(0, [3, 5], 0.20, [1, 0]), _rec(1, [3, 5], 0.05, [1, 0]),
               _rec(2, [4, 4], 0.20, [1, 0]), _rec(3, [1, 2], 0.50, [1, 0])]
    recs, picks = af.label_unknown(main, swapped)
    kept = {r["release"]: (r["kept_order"], r["final_loss"]) for r in recs}
    assert kept[0] == ("true", 0.10) and kept[1] == ("swapped", 0.05)
    assert kept[2][0] == "true"  # a tie keeps the true order
    assert kept[3] == ("swapped", 0.50)  # NaN counts as +inf
    assert all(r["attacker"] == af.NOMAP for r in recs)
    assert [p["true_kept"] for p in picks] == [True, False, False]  # release 2 has equal labels


def test_label_unknown_refuses_mismatched_releases():
    with pytest.raises(Exception):
        af.label_unknown([_rec(0, [3, 5], 0.1)], [dict(_rec(0, [3, 5], 0.2, [1, 0]), checkpoint=2)])


def test_a_cell_without_all_releases_is_not_run():
    cells = {("mnist-fresh", 2, 1, "1x2", "cosine"): {r: {} for r in range(30)},
             ("mnist-fresh", 2, 5, "1x2", af.R8): {r: {} for r in range(12)}}
    r = af.row(cells, {}, "mnist-fresh", 2, af.R8)
    assert r["outcome"] == "not run" and len(r["missing"]) == 2
