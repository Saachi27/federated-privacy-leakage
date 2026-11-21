"""Tests for the pilot timing projection (scripts/pilot_timing.py)."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import attack_multiclient as am
from scripts import pilot_timing as pt


def _times(n, sec):
    return {pt.cell_key(e, s, a): [sec] for e, s, a in am.cells(n, [1, 5], ["single", "multi"], am.ATTACKERS)}


def test_projection_scales_to_30_releases_and_applies_the_cut_order():
    times = {("mnist-fresh", 2): _times(2, 10.0), ("mnist-fresh", 4): _times(4, 20.0)}
    rows, levels = pt.project(times, workers=4, packing=1.0, night_hours=10)
    by = {(r["setting"], r["n"]): r for r in rows}
    assert by[("mnist-fresh", 2)]["per_release"] == 80.0  # 8 cells x 10 s
    assert by[("mnist-trained", 2)]["source"].startswith("= MNIST fresh")
    assert by[("mnist-fresh", 4)]["worker_hours"] == pytest.approx(160 * 30 / 3600)
    full = levels[0]["worker_hours"]
    assert full == pytest.approx(2 * (80 + 160) * 30 / 3600)
    assert levels[3]["worker_hours"] == pytest.approx(80 * 30 / 3600)  # only fresh N = 2 left
    assert levels[3]["wall_hours"] == pytest.approx(levels[3]["worker_hours"] / 4)
    assert all(lv["fits"] for lv in levels)


def test_missing_n4_timing_is_reported_not_guessed():
    rows, levels = pt.project({("mnist-fresh", 2): _times(2, 10.0)}, 4, 1.0, 10)
    assert levels[0]["worker_hours"] is None and levels[2]["worker_hours"] is not None
