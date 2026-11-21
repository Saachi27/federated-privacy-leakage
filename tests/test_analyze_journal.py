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


def _write_world(tmp, ssim_of, hit_cells=(), setting="mnist-fresh", drop=()):
    """Fake records for 30 releases of N = 2 and a matching baseline file."""
    imgs = _gallery_images()
    labels = np.arange(len(imgs)) % 4
    rng = np.random.default_rng(1)
    for r in range(30):
        if r in drop:
            continue
        idx = [2 * r, 2 * r + 1]
        for e, split, kind in CELLS:
            s = [float(np.clip(ssim_of(e, split, kind) + rng.normal(0, 0.03), 0, 1)) for _ in idx]
            d = os.path.join(tmp, "records", setting, "N2")
            os.makedirs(d, exist_ok=True)
            name = f"E{e}_{split}_{kind}_r{r:02d}"
            slots = imgs[idx] if (e, split, kind) in hit_cells else torch.zeros(2, 1, HW, HW)
            np.savez_compressed(os.path.join(d, name + "_recon.npz"), recon_slots=slots.numpy().astype(np.float16))
            rec = {"setting": setting, "n": 2, "e": e, "split": [int(c) for c in split.split("x")],
                   "split_name": split, "attacker": kind, "release": r, "checkpoint": r // 10,
                   "image_indices": idx, "labels": labels[idx].tolist(), "slot_labels": labels[idx].tolist(),
                   "ssim": s, "drift_cosine": 1.0 if e == 1 else 0.85 + 0.001 * r, "norm_ratio": 1.1,
                   "recon_file": name + "_recon.npz"}
            with open(os.path.join(d, name + ".json"), "w") as f:
                json.dump(rec, f)
    base = {"releases": {"2": [{"release": r, "b": 0.3, "A": 0.25, "S": 0.3, "baseline_any_pass": 0.0,
                                "baseline_hit": 0.0, "hit_exact_chance": 0.05,
                                "context_mean_ssim": {"all black": 0.01}} for r in range(30)]}}
    with open(os.path.join(tmp, "baselines.json"), "w") as f:
        json.dump(base, f)
    return rr.Gallery(imgs, labels, np.arange(len(imgs)))


def _run(tmp, gallery, objective="cosine"):
    records = aj.load_records(os.path.join(tmp, "records"))
    baselines, _ = aj.load_baselines(os.path.join(tmp, "baselines.json"))
    return aj.analyze(records, baselines, gallery, objective)


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
