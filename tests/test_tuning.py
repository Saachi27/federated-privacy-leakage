"""Tests for the attack-tuning grid (scripts/tune_attack.py)."""

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.tune_attack import OLD_TVS, TVS, grid_edge


def test_old_best_tv_is_no_longer_at_the_edge():
    # 0.01 won on mnist-trained and cifar10-fresh at the top of the old grid.
    assert grid_edge(OLD_TVS, 1e-2) == "high"
    assert grid_edge(TVS, 1e-2) is None
    assert max(TVS) >= 10 * max(OLD_TVS)  # at least one decade wider
    assert set(OLD_TVS) <= set(TVS)  # old weights still on the grid
    assert grid_edge(TVS, 1e-4) == "low" and grid_edge(TVS, 1e-3) is None


def test_old_tuning_files_used_the_old_grid():
    for path in glob.glob(os.path.join(ROOT, "results", "study", "*", "tuning.json")):
        with open(path) as f:
            assert json.load(f)["args"]["tvs"] == OLD_TVS, path
