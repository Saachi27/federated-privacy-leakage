"""Why N x 1 leaks more than 1 x N: how far each E = 5 release moves from the gradient."""

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.attack_multiclient as am  # noqa: E402
from src import multiclient as mc  # noqa: E402
from src.settings import fedavg_update  # noqa: E402

TOL = 1e-4


def load(path):
    with open(path) as f:
        return json.load(f)


def releases(records_dir, setting, n):
    """Release id -> (E = 1 record, {split name: attacker 2 SSIM mean}) for one cell."""
    out = {}
    for path in sorted(glob.glob(os.path.join(records_dir, setting, f"N{n}", f"E1_1x{n}_cosine_r*.json"))):
        rec = load(path)
        out[rec["release"]] = {"e1": rec}
    for name in (f"1x{n}", f"{n}x1"):
        for path in glob.glob(os.path.join(records_dir, setting, f"N{n}", f"E5_{name}_unrolled_r*.json")):
            rec = load(path)
            out[rec["release"]][name] = rec
    return out


def one_release(rec_e1, rec_multi, setting):
    images, labels, meta = am.load_images("mnist", rec_e1["image_indices"])
    model, h = am.load_checkpoint(setting, meta, rec_e1["checkpoint"])
    assert h == rec_e1["checkpoint_hash"], "checkpoint hash differs from the saved record"
    n, lr = len(images), rec_e1["local_lr"]
    g = mc.make_release(model, images, labels, (1, n), 1, lr)
    multi = mc.make_release(model, images, labels, (n, 1), 5, lr)
    single = mc.make_release(model, images, labels, (1, n), 5, lr)
    per_client = []
    for i in range(n):
        x, y = images[i:i + 1], labels[i:i + 1]
        per_client.append(mc.cosine(fedavg_update(model, x, y, 5, lr), fedavg_update(model, x, y, 1, lr)))
    drift = mc.cosine(multi, single)
    saved = rec_multi["drift_cosine"]
    if abs(drift - saved) > TOL:
        sys.exit(f"{setting} release {rec_e1['release']}: drift {drift:.6f} != saved {saved:.6f}; stopping")
    return {
        "cos_multi_g": mc.cosine(multi, g),
        "cos_single_g": mc.cosine(single, g),
        "min_client_cos": min(per_client),
        "drift": drift,
        "saved_drift": saved,
    }


def summarize(rows, key):
    v = np.array([r[key] for r in rows])
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", default="results/journal/targets")
    p.add_argument("--out", default="results/journal/why_more")
    args = p.parse_args()

    out = {"note": "post hoc, descriptive; not a locked endpoint", "cells": []}
    for setting in ("mnist-fresh", "mnist-trained"):
        for n in (2, 4):
            rows = []
            for rid, r in sorted(releases(args.records, setting, n).items()):
                row = one_release(r["e1"], r[f"{n}x1"], setting)
                row.update({
                    "release": rid,
                    "checkpoint": r["e1"]["checkpoint"],
                    "labels": r["e1"]["labels"],
                    "ssim_multi": float(np.mean(r[f"{n}x1"]["ssim"])),
                    "ssim_single": float(np.mean(r[f"1x{n}"]["ssim"])),
                })
                rows.append(row)
            cell = {"setting": setting, "n": n, "releases": len(rows), "rows": rows}
            for key in ("cos_multi_g", "cos_single_g", "min_client_cos", "drift"):
                cell[key] = summarize(rows, key)
            gap = np.array([r["ssim_multi"] - r["ssim_single"] for r in rows])
            drift = np.array([r["drift"] for r in rows])
            cell["spearman_drift_vs_gap"] = float(_spearman(drift, gap))
            out["cells"].append(cell)
            print(f"{setting} N={n}: cos(Nx1,g) {cell['cos_multi_g']['median']:.3f}, "
                  f"cos(1xN,g) {cell['cos_single_g']['median']:.3f}, drift check ok ({len(rows)} releases)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(out, f, indent=1)
    with open(args.out + ".md", "w") as f:
        f.write(markdown(out))
    print(f"saved {args.out}.json, {args.out}.md")


def _spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def markdown(out):
    lines = [
        "# Why N × 1 leaks more (post hoc)",
        "",
        "*Made by `scripts/why_more.py` from the saved target releases. Descriptive only: not a locked "
        "endpoint, changes no outcome. No attack was rerun.*",
        "",
        "g = the E = 1 release (mean gradient). Cosines are over all parameters. "
        "Every recomputed drift matched the saved `drift_cosine` within 1e-4.",
        "",
        "| Setting | N | Releases | cos(N×1 E=5, g) median [min] | cos(1×N E=5, g) median [min] | "
        "Worst one-image client cos(δ_c, g_c) median | Spearman(drift, SSIM gap) |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in out["cells"]:
        lines.append(
            f"| {c['setting']} | {c['n']} | {c['releases']} | {c['cos_multi_g']['median']:.3f} "
            f"[{c['cos_multi_g']['min']:.3f}] | {c['cos_single_g']['median']:.3f} [{c['cos_single_g']['min']:.3f}] | "
            f"{c['min_client_cos']['median']:.3f} | {c['spearman_drift_vs_gap']:.2f} |")
    lines += [
        "",
        "Reading: a one-image client's five local steps keep its update almost parallel to its own "
        "gradient (every client cos ≥ 0.99), so the N × 1 secure sum points where the E = 1 batch "
        "gradient points. For a direction-matching attacker that release is as exposed as E = 1, which "
        "matches its SSIM (MNIST fresh N = 2: 0.855 at E = 5 vs 0.859 at E = 1). One client training on "
        "all N images moves away from g, because the images interact through the shared weights.",
        "",
        "Limits of this reading: across releases, how far 1 × N moves from g only weakly predicts the "
        "size of the SSIM gap (Spearman column; the sign is mixed). On MNIST trained at N = 2 the 1 × N "
        "release stays close to g (median 0.986) yet attacker 2 scores 0.355 on it, below attacker 1's "
        "0.531, so part of the trained-model gap is attacker 2 finding the interacting 1 × N problem harder.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
