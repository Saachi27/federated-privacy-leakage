"""End-to-end test of scripts/analyze_journal.py on fake records (no real target data)."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import analyze_journal as aj
from src import release_risk as rr, stats

HW = 12
CELLS = [(1, "1x2", "lbfgs"), (1, "1x2", "cosine")] + \
        [(5, s, k) for s in ("1x2", "2x1") for k in ("lbfgs", "cosine", "unrolled")]


def _gallery_images(n=80):
    g = torch.Generator().manual_seed(0)
    return torch.nn.functional.avg_pool2d(torch.rand(n, 1, HW, HW, generator=g), 3, 1, 1)


def _write_world(tmp, ssim_of, hit_cells=(), setting="mnist-fresh", drop=(), drop_cells=None,
                 same_label=(), ssim_r=None, skip_cells=()):
    """Fake records for 30 releases of N = 2 and a matching baseline file."""
    drop_cells = drop_cells or {}
    imgs = _gallery_images()
    labels = np.arange(len(imgs)) % 4
    rng = np.random.default_rng(1)
    for r in range(30):
        if r in drop:
            continue
        idx = [2 * r, 2 * r + 1]
        for e, split, kind in CELLS:
            if r in drop_cells.get((e, split, kind), ()) or (e, split, kind) in skip_cells:
                continue
            mu = ssim_r(e, split, kind, r) if ssim_r else ssim_of(e, split, kind)
            s = [float(np.clip(mu + rng.normal(0, 0.03), 0, 1)) for _ in idx]
            rel_labels = [int(labels[idx[0]])] * 2 if r in same_label else labels[idx].tolist()
            d = os.path.join(tmp, "records", setting, "N2")
            os.makedirs(d, exist_ok=True)
            name = f"E{e}_{split}_{kind}_r{r:02d}"
            slots = imgs[idx] if (e, split, kind) in hit_cells else torch.zeros(2, 1, HW, HW)
            np.savez_compressed(os.path.join(d, name + "_recon.npz"), recon_slots=slots.numpy().astype(np.float16))
            rec = {"setting": setting, "n": 2, "e": e, "split": [int(c) for c in split.split("x")],
                   "split_name": split, "attacker": kind, "release": r, "checkpoint": r // 10,
                   "checkpoint_hash": f"hash-{setting}-{r // 10}",
                   "image_indices": idx, "labels": rel_labels, "slot_labels": labels[idx].tolist(),
                   "ssim": s, "drift_cosine": 1.0 if e == 1 else 0.85 + 0.001 * r, "norm_ratio": 1.1,
                   "recon_file": name + "_recon.npz"}
            with open(os.path.join(d, name + ".json"), "w") as f:
                json.dump(rec, f)
    base = {"releases": {"2": [{"release": r, "image_indices": [2 * r, 2 * r + 1], "b": 0.3, "A": 0.25, "S": 0.3, "baseline_any_pass": 0.0,
                                "baseline_hit": 0.0, "hit_exact_chance": 0.05,
                                "context_mean_ssim": {"all black": 0.01}} for r in range(30)]},
            "gallery_sizes": {str(c): int((labels == c).sum()) for c in range(4)}}
    with open(os.path.join(tmp, "baselines.json"), "w") as f:
        json.dump(base, f)
    return rr.Gallery(imgs, labels, np.arange(len(imgs)))


def _run(tmp, gallery, objective="cosine"):
    records = aj.load_records(os.path.join(tmp, "records"))
    baselines, data = aj.load_baselines(os.path.join(tmp, "baselines.json"))
    return aj.analyze(records, baselines, gallery, objective, data.get("gallery_sizes"))


def _scenario(gap):
    def ssim_of(e, split, kind):
        if e == 1:
            return 0.8
        if kind == "unrolled":
            return 0.5 + (gap if split == "2x1" else 0.0)
        return 0.45
    return ssim_of


def test_main_outcome_more_with_retrieval(tmp_path):
    gallery = _write_world(str(tmp_path), _scenario(0.25), hit_cells={(5, "2x1", "unrolled")})
    res = _run(str(tmp_path), gallery)
    assert res["main"]["outcome"] == stats.MORE
    assert res["main"]["pc_passes"] and res["main"]["multi_bb"]
    assert res["main"]["gap"]["n"] == 30 and res["main"]["gap"]["t90"]["lo"] > 0.05
    multi = res["cells"]["mnist-fresh|2|5|2x1|unrolled"]
    single = res["cells"]["mnist-fresh|2|5|1x2|unrolled"]
    # Exact copies always hit; blank images only by chance in this small fake gallery.
    assert multi["hit_rate"] == 1.0 and single["hit_rate"] < 0.5
    counts = res["gaps"]["mnist-fresh|2|unrolled"]["hit"]["counts"]
    assert counts["only multi"] + counts["both"] == 30 and counts["only single"] == counts["neither"] == 0
    assert res["gaps"]["mnist-fresh|2|unrolled"]["hit"]["t90"]["lo"] > 0
    assert res["drift"]["mnist-fresh|2"]["n"] == 30 and res["drift"]["mnist-fresh|2"]["e1_min"] == 1.0
    md = aj.render(res)
    assert "**Outcome: more**" in md and "gave better reconstructions than" in md
    json.dumps(aj._jsonable(res))


@pytest.mark.parametrize("gap,want", [(0.0, stats.SAME), (-0.25, stats.LESS)])
def test_main_outcome_same_and_less(tmp_path, gap, want):
    gallery = _write_world(str(tmp_path), _scenario(gap))
    assert _run(str(tmp_path), gallery)["main"]["outcome"] == want


def test_not_tested_when_the_positive_control_fails(tmp_path):
    def ssim_of(e, split, kind):
        return 0.2 if e == 1 else 0.6
    gallery = _write_world(str(tmp_path), ssim_of)
    res = _run(str(tmp_path), gallery)
    assert res["main"]["outcome"] == stats.NOT_TESTED


def test_neither_above_baseline(tmp_path):
    def ssim_of(e, split, kind):
        return 0.8 if e == 1 else 0.25
    gallery = _write_world(str(tmp_path), ssim_of)
    assert _run(str(tmp_path), gallery)["main"]["outcome"] == stats.NEITHER


def test_basic_main_objective_uses_the_lbfgs_positive_control(tmp_path):
    def ssim_of(e, split, kind):
        if e == 1:
            return 0.8 if kind == "cosine" else 0.1  # basic fails at E = 1
        return 0.5
    gallery = _write_world(str(tmp_path), ssim_of)
    assert _run(str(tmp_path), gallery, "cosine")["main"]["outcome"] == stats.SAME
    assert _run(str(tmp_path), gallery, "basic")["main"]["outcome"] == stats.NOT_TESTED


def test_too_few_releases_on_a_checkpoint_gives_no_label(tmp_path):
    gallery = _write_world(str(tmp_path), _scenario(0.25), drop=range(24, 30))  # checkpoint 2 keeps 4
    assert _run(str(tmp_path), gallery)["main"]["outcome"] == stats.DESCRIPTIVE


def test_command_line_end_to_end(tmp_path):
    _write_world(str(tmp_path), _scenario(0.0))
    out = str(tmp_path / "report")
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "analyze_journal.py"),
                    "--records", str(tmp_path / "records"), "--baselines", str(tmp_path / "baselines.json"),
                    "--main-objective", "cosine", "--no-retrieval", "--out", out],
                   check=True, capture_output=True, cwd=ROOT)
    with open(out + ".md") as f:
        md = f.read()
    assert "**Outcome: same**" in md and "## 6. Secondary objective (attacker 1, basic)" in md
    with open(out + ".json") as f:
        assert json.load(f)["main"]["outcome"] == "same"


MULTI2 = (5, "2x1", "unrolled")
SINGLE2 = (5, "1x2", "unrolled")


def test_sparse_gap_is_descriptive_with_no_pass_fail(tmp_path):
    # Only the 2 x 1 cell misses releases: the gap has 4 paired releases on checkpoint 2.
    gallery = _write_world(str(tmp_path), _scenario(0.25), drop_cells={MULTI2: range(24, 30)})
    m = _run(str(tmp_path), gallery)["main"]
    assert m["outcome"] == stats.DESCRIPTIVE
    assert m["gap"]["status"] == stats.DESCRIPTIVE and m["gap"]["t90"] is None
    assert m["gap"]["per_checkpoint_n"] == {0: 10, 1: 10, 2: 4}
    assert m["gap"]["boot90"]["n"] == 24 and m["gap"]["mean_diff"] > 0.05  # the numbers are still shown
    assert m["pc_passes"] is None and m["single_bb"] is None and m["multi_bb"] is None
    md = aj.render(_run(str(tmp_path), gallery))
    assert "**Outcome: descriptive only**" in md and "descriptive only (no pass/fail)" in md
    assert "passes |" not in md.split("## 2.")[0] and "k2: 4" in md


def test_sparse_positive_control_gets_no_pass_fail(tmp_path):
    gallery = _write_world(str(tmp_path), _scenario(0.25), drop_cells={(1, "1x2", "cosine"): range(0, 7)})
    m = _run(str(tmp_path), gallery)["main"]
    assert m["outcome"] == stats.DESCRIPTIVE and m["pc"]["t95"] is None and m["pc_passes"] is None


def test_cells_with_no_shared_release_report_no_paired_releases(tmp_path):
    gallery = _write_world(str(tmp_path), _scenario(0.25),
                           drop_cells={MULTI2: range(0, 15), SINGLE2: range(15, 30)})
    res = _run(str(tmp_path), gallery)
    m = res["main"]
    assert m["outcome"] == stats.DESCRIPTIVE
    assert m["gap"]["n"] == 0 and m["gap"]["status"] == stats.NO_PAIRS and m["gap"]["mean_diff"] is None
    assert res["gaps"]["mnist-fresh|2|unrolled"]["hit"]["status"] == stats.NO_PAIRS
    md = aj.render(res)
    assert "no paired releases" in md
    json.dumps(aj._jsonable(res))


def _edit_record(tmp, cell, r, **change):
    e, split, kind = cell
    path = os.path.join(tmp, "records", "mnist-fresh", "N2", f"E{e}_{split}_{kind}_r{r:02d}.json")
    with open(path) as f:
        rec = json.load(f)
    rec.update(change)
    with open(path, "w") as f:
        json.dump(rec, f)


@pytest.mark.parametrize("change", [{"image_indices": [2, 99]}, {"checkpoint": 2},
                                    {"checkpoint_hash": "another model"}])
def test_paired_records_that_differ_stop_the_analysis(tmp_path, change):
    gallery = _write_world(str(tmp_path), _scenario(0.0))
    _edit_record(str(tmp_path), MULTI2, 1, **change)
    with pytest.raises(aj.PairingError):
        _run(str(tmp_path), gallery)


def test_a_record_without_checkpoint_hash_stops_the_analysis(tmp_path):
    gallery = _write_world(str(tmp_path), _scenario(0.0))
    path = os.path.join(str(tmp_path), "records", "mnist-fresh", "N2", "E5_2x1_unrolled_r03.json")
    with open(path) as f:
        rec = json.load(f)
    del rec["checkpoint_hash"]
    with open(path, "w") as f:
        json.dump(rec, f)
    with pytest.raises(aj.PairingError):
        _run(str(tmp_path), gallery)


def test_check_pair_catches_a_different_release_id():
    a = {"release": 3, "image_indices": [1, 2], "checkpoint": 0, "checkpoint_hash": "h"}
    with pytest.raises(aj.PairingError):
        aj.check_pair(a, dict(a, release=4), "gap")
    aj.check_pair(a, dict(a), "gap")


def test_baselines_are_joined_by_image_indices_too(tmp_path):
    gallery = _write_world(str(tmp_path), _scenario(0.0))
    path = os.path.join(str(tmp_path), "baselines.json")
    with open(path) as f:
        base = json.load(f)
    base["releases"]["2"][5]["image_indices"] = [10, 12]  # release 5 is images 10, 11
    with open(path, "w") as f:
        json.dump(base, f)
    with pytest.raises(aj.PairingError, match="image_indices"):
        _run(str(tmp_path), gallery)
    del base["releases"]["2"][5]["image_indices"]
    with open(path, "w") as f:
        json.dump(base, f)
    with pytest.raises(aj.PairingError, match="no image_indices"):
        aj.load_baselines(path)


SAME_RELEASES = [r for r in range(30) if r % 10 < 6]  # 6 per checkpoint; 4 per checkpoint differ


def test_decision_18_splits_the_main_gap_by_labels_and_lists_drift(tmp_path):
    def ssim_r(e, split, kind, r):  # gap 0 for same-label releases, 0.25 otherwise
        base = _scenario(0.0)(e, split, kind)
        return base + (0.25 if (e, split, kind) == MULTI2 and r not in SAME_RELEASES else 0.0)
    _write_world(str(tmp_path), None, ssim_r=ssim_r, same_label=SAME_RELEASES)
    res = _run(str(tmp_path), None)
    same, diff = res["label_split"][aj.SAME_LABEL], res["label_split"][aj.DIFF_LABEL]
    assert same["releases"] == SAME_RELEASES and same["status"] == "decides"
    assert abs(same["t90"]["mean"]) < 0.03 and same["per_checkpoint_n"] == {0: 6, 1: 6, 2: 6}
    assert diff["n"] == 12 and diff["status"] == stats.DESCRIPTIVE and diff["t90"] is None
    assert diff["mean_diff"] > 0.2  # numbers only, no interval that decides
    rows = res["drift_releases"]["mnist-fresh|2"]
    assert [r["release"] for r in rows] == list(range(30))
    assert [r["release"] for r in rows if r["same_label"]] == SAME_RELEASES
    assert rows[7]["drift"] == pytest.approx(0.857)
    md = aj.render(res)
    assert "## 8. Drift and labels (pre-declared" in md and "| same label | 18 |" in md
    assert "| different labels | 12 |" in md and "| 7 | 0 | 2, 3 | no | 0.8570 |" in md


def test_decision_18_never_changes_the_main_outcome(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write_world(str(a), _scenario(0.25))
    _write_world(str(b), _scenario(0.25), same_label=SAME_RELEASES)
    ra, rb = _run(str(a), None), _run(str(b), None)
    assert ra["main"]["outcome"] == rb["main"]["outcome"] == stats.MORE
    assert ra["main"]["gap"]["t90"] == rb["main"]["gap"]["t90"]
    assert ra["label_split"][aj.SAME_LABEL]["n"] == 0  # all pairs differ in world a
    assert ra["label_split"][aj.SAME_LABEL]["status"] == stats.NO_PAIRS


def test_report_lists_missing_release_ids_per_cell(tmp_path):
    gallery = _write_world(str(tmp_path), _scenario(0.0), drop=[29],
                           drop_cells={MULTI2: [3, 17]}, skip_cells={(5, "2x1", "lbfgs")})
    res = _run(str(tmp_path), gallery)
    miss = res["missing_releases"]
    assert miss["mnist-fresh|2|5|2x1|unrolled"] == {"expected": 30, "have": 27, "missing": [3, 17, 29]}
    assert miss["mnist-fresh|2|5|1x2|unrolled"]["missing"] == [29]
    assert miss["mnist-fresh|2|5|2x1|lbfgs"]["have"] == 0 and len(miss["mnist-fresh|2|5|2x1|lbfgs"]["missing"]) == 30
    md = aj.render(res)
    assert "## 9. Releases per cell" in md
    assert "| mnist-fresh | 2 | 5 | 2x1 | attacker 2, cosine | 27 | 30 | 3, 17, 29 |" in md
    assert "| mnist-fresh | 2 | 5 | 2x1 | attacker 1, basic | 0 | 30 | all (no records) |" in md


def test_retrieval_section_reports_every_class_gallery_size(tmp_path):
    gallery = _write_world(str(tmp_path), _scenario(0.0), hit_cells={MULTI2})
    res = _run(str(tmp_path), gallery)
    assert res["gallery_sizes"] == {0: 20, 1: 20, 2: 20, 3: 20}
    md = aj.render(res)
    retrieval = md.split("## 7. Retrieval")[1].split("## 8.")[0]
    assert "class 0: 20, class 1: 20, class 2: 20, class 3: 20" in retrieval
    # Without a gallery, the sizes come from the baseline file.
    assert "class 3: 20" in aj.render(_run(str(tmp_path), None)).split("## 7. Retrieval")[1]


def test_gallery_that_disagrees_with_the_baseline_file_stops(tmp_path):
    gallery = _write_world(str(tmp_path), _scenario(0.0))
    small = rr.Gallery(_gallery_images(40), np.arange(40) % 4, np.arange(40))
    records = aj.load_records(os.path.join(str(tmp_path), "records"))
    baselines, data = aj.load_baselines(os.path.join(str(tmp_path), "baselines.json"))
    with pytest.raises(ValueError, match="gallery sizes"):
        aj.analyze(records, baselines, small, "cosine", data["gallery_sizes"])


def test_command_line_end_to_end_with_every_new_section(tmp_path):
    """Fake records only: sparse cell, missing ids, label split and gallery sizes in one report."""
    _write_world(str(tmp_path), _scenario(0.25), same_label=SAME_RELEASES, drop_cells={MULTI2: range(24, 30)})
    out = str(tmp_path / "report")
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "analyze_journal.py"),
                    "--records", str(tmp_path / "records"), "--baselines", str(tmp_path / "baselines.json"),
                    "--main-objective", "cosine", "--no-retrieval", "--out", out],
                   check=True, capture_output=True, cwd=ROOT)
    with open(out + ".md") as f:
        md = f.read()
    assert "**Outcome: descriptive only**" in md and "descriptive only (no pass/fail)" in md
    assert "## 8. Drift and labels" in md and "| same label |" in md
    assert "24, 25, 26, 27, 28, 29 |" in md and "class 0: 20" in md
    with open(out + ".json") as f:
        res = json.load(f)
    assert res["main"]["outcome"] == stats.DESCRIPTIVE and res["main"]["pc_passes"] is None
    assert res["missing_releases"]["mnist-fresh|2|5|2x1|unrolled"]["missing"] == [24, 25, 26, 27, 28, 29]
