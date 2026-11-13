"""No-information baselines: what SSIM a "reconstruction" gets without any leak."""

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.attack_sum import match_images
from src import metrics
from src.data import load_datasets

FIRST, NUM = 50, 32  # the K sweep's images


def mean_ci(per_seed):
    """Mean over seeds and the 95% CI half-width (t distribution)."""
    a = np.asarray(per_seed, dtype=float)
    if len(a) < 2:
        return float(a.mean()), float("nan")
    return float(a.mean()), float(stats.t.ppf(0.975, len(a) - 1) * stats.sem(a))


def score(fakes, targets):
    recs = [metrics.all_metrics(f, t) for f, t in zip(fakes, targets)]
    return {m: float(np.mean([r[m] for r in recs])) for m in ("ssim", "psnr", "mse")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--out", default="results/baselines")
    args = p.parse_args()
    torch.set_num_threads(2)

    train_ds, test_ds, _ = load_datasets("mnist", normalize=False)
    targets = [test_ds[i][0] for i in range(FIRST, FIRST + NUM)]
    pool_idx = np.arange(FIRST + NUM, len(test_ds))  # never a target, never a tuning image
    average = (train_ds.data.float() / 255.0).mean(0, keepdim=True)

    rows = {}  # name -> {metric: [per-seed values]}

    def add(name, seed_scores):
        rows[name] = {m: [s[m] for s in seed_scores] for m in ("ssim", "psnr", "mse")}

    add("all black", [score([torch.zeros_like(t) for t in targets], targets)])
    add("average digit", [score([average for _ in targets], targets)])
    noise = []
    for seed in args.seeds:
        g = torch.Generator().manual_seed(seed)
        noise.append(score([torch.rand(t.shape, generator=g) for t in targets], targets))
    add("random noise", noise)

    for k in args.ks:
        per_seed = []
        for seed in args.seeds:
            # Same grouping as scripts/attack_sum.py make_tasks.
            order = np.random.default_rng(seed).permutation(NUM)
            rng = np.random.default_rng(1000 + seed)
            recs = []
            for group in range(NUM // k):
                chosen = [targets[int(i)] for i in order[group * k:(group + 1) * k]]
                others = torch.stack([test_ds[int(i)][0] for i in rng.choice(pool_idx, k, replace=False)])
                recs += match_images(others, chosen)
            per_seed.append({m: float(np.mean([r[m] for r in recs])) for m in ("ssim", "psnr", "mse")})
        add(f"unrelated digits, best-matched, K={k}", per_seed)

    summary = {}
    for name, vals in rows.items():
        summary[name] = {m: dict(zip(("mean", "ci95"), mean_ci(v))) for m, v in vals.items()}
        summary[name]["seeds"] = len(vals["ssim"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump({"images": f"MNIST test {FIRST}-{FIRST + NUM - 1}", "seeds": args.seeds,
                   "baselines": summary, "per_seed": rows}, f, indent=2)

    def fmt(d):
        return f"{d['mean']:.3f}" if np.isnan(d["ci95"]) else f"{d['mean']:.3f} ± {d['ci95']:.3f}"

    lines = [f"# No-information baselines (MNIST test images {FIRST}-{FIRST + NUM - 1})", "",
             "A reconstruction must score clearly above these to show a leak. "
             "± is the 95% CI over seeds (deterministic rows have none).", "",
             "| Fake reconstruction | SSIM | PSNR (dB) | MSE |", "|---|---|---|---|"]
    for name, s in summary.items():
        lines.append(f"| {name} | {fmt(s['ssim'])} | {fmt(s['psnr'])} | {fmt(s['mse'])} |")
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved -> {args.out}.json, {args.out}.md")


if __name__ == "__main__":
    main()
