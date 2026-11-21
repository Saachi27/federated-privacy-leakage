"""Tests for the client-split runner (scripts/attack_multiclient.py): releases, cells and records."""

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import attack_multiclient as am
from src import splits

HAS_DATA = os.path.isdir(os.path.join(ROOT, "data"))
HAS_SPLIT_FILE = os.path.exists(os.path.join(ROOT, splits.SPLIT_FILE.format(dataset="mnist")))

# What each target record must save.
RECORD_FIELDS = ["setting", "n", "e", "split", "attacker", "objective", "checkpoint", "checkpoint_hash",
                 "release", "image_indices", "labels", "client_partition", "local_lr", "tv",
                 "restart_seeds", "restart_losses", "best_restart", "ssim", "slot_to_target",
                 "slot_labels", "psnr", "mse", "ssim_zeropad", "drift_cosine", "norm_ratio", "seconds"]


@pytest.fixture
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.mark.skipif(not HAS_SPLIT_FILE, reason="split file not written")
@pytest.mark.parametrize("n", [2, 4])
def test_target_releases_follow_the_plan(in_root, n):
    rel = am.target_releases(n)
    assert len(rel) == 30
    assert [r["image_indices"] for r in rel] == splits.JOURNAL.releases(n)
    assert [r["checkpoint"] for r in rel] == [0] * 10 + [1] * 10 + [2] * 10
    assert [r["release"] for r in rel] == list(range(30))


def test_restart_seeds_follow_the_plan():
    assert am.restart_seed(2, 0) == 20000 and am.restart_seed(4, 29) == 40290


def test_pilot_releases_use_tuning_images_only():
    rel = am.pilot_releases(2)
    assert len(rel) == 25 and rel[3]["image_indices"] == [6, 7]
    assert [r["checkpoint"] for r in rel[:4]] == [0, 1, 2, 0]
    assert len(am.pilot_releases(4)) == 12 and am.pilot_releases(4)[1]["image_indices"] == [4, 5, 6, 7]
    used = {i for r in rel + am.pilot_releases(4) for i in r["image_indices"]}
    assert used <= set(splits.JOURNAL.tuning)


def test_smoke_range_may_not_touch_a_target():
    with pytest.raises(AssertionError):
        am.range_releases(2, 5, (82, 400), 0)
    rel = am.range_releases(2, 3, (50, 82), 0)  # retired images: allowed for a smoke test
    assert len(rel) == 3 and all(50 <= i < 82 for r in rel for i in r["image_indices"])


def test_cells_and_tv_keys():
    c = am.cells(2, [1, 5], ["single", "multi"], am.ATTACKERS)
    assert (1, (1, 2), "unrolled") not in c and (1, (2, 1), "cosine") not in c
    assert len(c) == 2 + 2 * 3
    assert am.tv_key(1, "cosine") == "T-A" and am.tv_key(5, "cosine") == "T-B"
    assert am.tv_key(5, "unrolled") == "T-C" and am.tv_key(5, "lbfgs") is None


def test_lr_rule_picks_the_smallest_passing_lr_or_the_edge():
    assert am.pick_lr({0.01: 0.99, 0.03: 0.95, 0.1: 0.89, 0.3: 0.7}) == (0.1, False)
    assert am.pick_lr({0.01: 0.99, 0.03: 0.98, 0.1: 0.97, 0.3: 0.95}) == (0.3, True)
    assert am.pick_lr({0.01: 0.9, 0.3: 0.5}) == (0.01, False)  # <= 0.9 passes
    pairs = am.lr_tuning_pairs()
    assert len(pairs) == 25 and pairs[0] == [0, 1] and pairs[24] == [48, 49]


def _args(tmp_path, *extra):
    return am.parse_args(["--releases", "pilot", "--settings", "mnist-fresh", "--ns", "2",
                          "--local-lr", "0.3", "--cosine-tv", "1e-3", "--iters", "1", "--cosine-iters", "2",
                          "--unrolled-iters", "2", "--out", str(tmp_path), *extra])


def test_task_list_pairs_every_cell_on_the_same_release(tmp_path):
    tasks = am.make_tasks(_args(tmp_path, "--groups", "2"))
    assert len(tasks) == 2 * 8
    for r in (0, 1):
        same = [t for t in tasks if t["release"] == r]
        assert len({tuple(t["image_indices"]) for t in same}) == 1
        assert {tuple(t["restart_seeds"]) for t in same} == {(20000 + 10 * r, 20001 + 10 * r)}
        assert {t["checkpoint"] for t in same} == {r % 3}


@pytest.mark.skipif(not HAS_DATA, reason="MNIST not downloaded")
def test_record_saves_everything_section_16_lists_and_resume_checks_it(tmp_path, in_root):
    args = _args(tmp_path, "--groups", "1", "--workers", "1")
    tasks = am.make_tasks(args)
    unrolled = next(t for t in tasks if t["attacker"] == "unrolled" and t["split"] == [2, 1])
    rec = am.run_task(unrolled)
    for k in RECORD_FIELDS:
        assert k in rec, k
    assert rec["attacker_number"] == 2 and rec["objective"] == "cosine" and rec["tv"] == 1e-3
    assert rec["client_partition"] == [[0], [1]] and rec["restart_seeds"] == [20000, 20001]
    assert len(rec["restart_losses"]) == 2 and rec["final_loss"] == rec["restart_losses"][rec["best_restart"]]
    assert sorted(rec["slot_to_target"]) == [0, 1] and rec["slot_labels"] == rec["labels"]
    assert [rec["target_to_slot"][j] for j in rec["slot_to_target"]] == [0, 1]
    with open(unrolled["path"]) as f:
        assert json.load(f)["ssim"] == rec["ssim"]
    slots = np.load(unrolled["path"].replace(".json", "_recon.npz"))["recon_slots"]
    assert slots.shape == (2, 1, 28, 28) and slots.dtype == np.float16
    # Matched SSIM is recomputable from the slot-order images and the permutation.
    from src import metrics
    x, _, _ = am.load_images("mnist", rec["image_indices"])
    import torch
    for i, j in enumerate(rec["slot_to_target"]):
        s = metrics.ssim(torch.tensor(slots[i]).float(), x[j])
        assert s == pytest.approx(rec["ssim"][j], abs=2e-3)
    # E = 1 record: drift check passes and basic objective has no TV.
    e1 = next(t for t in tasks if t["e"] == 1 and t["attacker"] == "lbfgs")
    rec1 = am.run_task(e1)
    assert rec1["drift_cosine"] >= am.E1_DRIFT_MIN and rec1["tv"] is None and rec1["objective"] == "basic"
    # Resume: a finished record made with other settings stops the run.
    am.check_finished(unrolled)
    changed = dict(unrolled, tv=1e-2)
    with pytest.raises(SystemExit):
        am.check_finished(changed)


@pytest.mark.skipif(not HAS_DATA, reason="MNIST not downloaded")
def test_a_wrong_checkpoint_hash_is_fatal(tmp_path, in_root, monkeypatch):
    from src import checkpoints
    monkeypatch.setitem(checkpoints.FRESH_STATE_SHA256, 0, "0" * 64)
    t = am.make_tasks(_args(tmp_path, "--groups", "1"))[0]
    with pytest.raises(am.Fatal):
        am.run_task(t)
    assert "fatal" in am.safe_run_task(t)


def _fake_finished(task, checkpoint_hash):
    """A finished record for this task (as run_task would leave it), with a given hash."""
    os.makedirs(os.path.dirname(task["path"]), exist_ok=True)
    rec = {k: v for k, v in task.items() if k != "path"}
    rec["checkpoint_hash"] = checkpoint_hash
    with open(task["path"], "w") as f:
        json.dump(rec, f)
    np.savez_compressed(task["path"].replace(".json", "_recon.npz"), recon_slots=np.zeros((2, 1, 28, 28)))


def test_resume_stops_when_the_saved_checkpoint_hash_differs(tmp_path, monkeypatch):
    task = am.make_tasks(_args(tmp_path, "--groups", "1"))[0]
    monkeypatch.setattr(am, "current_checkpoint_hash", lambda setting, k: f"now-{setting}-{k}")
    _fake_finished(task, f"now-mnist-fresh-{task['checkpoint']}")
    am.check_finished(task)  # same hash: skipped quietly
    _fake_finished(task, "an older model")
    with pytest.raises(SystemExit, match="checkpoint hash"):
        am.check_finished(task)

    def broken(setting, k):
        raise am.Fatal("checkpoint 0 hash does not match the recorded one")
    monkeypatch.setattr(am, "current_checkpoint_hash", broken)
    with pytest.raises(SystemExit, match="STOP"):
        am.check_finished(task)


def test_current_checkpoint_hash_is_the_pinned_fresh_hash(monkeypatch):
    from src import checkpoints
    monkeypatch.setattr(am, "_dataset", lambda name: (None, {"in_channels": 1, "num_classes": 10,
                                                              "input_size": 28}))
    monkeypatch.setattr(am, "_HASHES", {})
    assert am.current_checkpoint_hash("mnist-fresh", 1) == checkpoints.FRESH_STATE_SHA256[1]
    monkeypatch.setitem(checkpoints.FRESH_STATE_SHA256, 2, "0" * 64)
    with pytest.raises(am.Fatal):
        am.current_checkpoint_hash("mnist-fresh", 2)
