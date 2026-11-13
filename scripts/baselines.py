"""No-information baselines: what a "reconstruction" scores without any leak."""

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

POOL_START = 82  # unrelated images come from here on
METRICS = ("ssim", "ssim_zeropad", "psnr", "mse")


def mean_ci(per_seed):
    """Mean over seeds and the 95% CI half-width (t distribution)."""
    a = np.asarray(per_seed, dtype=float)
    if len(a) < 2 or np.allclose(a, a[0]):
        return float(a.mean()), float("nan")
    return float(a.mean()), float(stats.t.ppf(0.975, len(a) - 1) * stats.sem(a))


def score(fakes, targets):
    recs = [metrics.all_metrics(f, t) for f, t in zip(fakes, targets)]
    return {m: float(np.mean([r[m] for r in recs])) for m in METRICS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p.add_argument("--first-image", type=int, default=50)
    p.add_argument("--num-images", type=int, default=32)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--out", default="results/baselines")
    args = p.parse_args()
    torch.set_num_threads(2)
    first, num = args.first_image, args.num_images
    assert first + num <= POOL_START

    train_ds, test_ds, _ = load_datasets(args.dataset, normalize=False)
    targets = [test_ds[i][0] for i in range(first, first + num)]
    labels = [int(test_ds[i][1]) for i in range(first, first + num)]
    test_labels = np.array([int(y) for y in test_ds.targets])
    train_labels = torch.as_tensor(np.array([int(y) for y in train_ds.targets]))
    pool_idx = np.arange(POOL_START, len(test_ds))
    pool_by_class = {c: pool_idx[test_labels[pool_idx] == c] for c in range(10)}
    train = torch.as_tensor(np.asarray(train_ds.data)).float() / 255.0
    train = train.unsqueeze(1) if train.dim() == 3 else train.permute(0, 3, 1, 2)  # (N, C, H, W)
    average = train.mean(0)
    class_avg = {c: train[train_labels == c].mean(0) for c in range(10)}

    rows = {}  # name -> {metric: [per-seed values]}

    def add(name, seed_scores):
        rows[name] = {m: [s[m] for s in seed_scores] for m in METRICS}

    add("all black", [score([torch.zeros_like(t) for t in targets], targets)])
    add("average digit", [score([average for _ in targets], targets)])
    add("class average (knows labels)", [score([class_avg[y] for y in labels], targets)])
    noise = []
    for seed in args.seeds:
        g = torch.Generator().manual_seed(seed)
        noise.append(score([torch.rand(t.shape, generator=g) for t in targets], targets))
    add("random noise", noise)

    for k in args.ks:
        unrelated, same_class = [], []
        for seed in args.seeds:
            order = np.random.default_rng(seed).permutation(num)  # same grouping as the experiments
            rng = np.random.default_rng(1000 + seed)
            u_recs, s_recs = [], []
            for group in range(num // k):
                idx = [int(i) for i in order[group * k:(group + 1) * k]]
                chosen = [targets[i] for i in idx]
                others = torch.stack([test_ds[int(i)][0] for i in rng.choice(pool_idx, k, replace=False)])
                u_recs += match_images(others, chosen)
                same = torch.stack([test_ds[int(rng.choice(pool_by_class[labels[i]]))][0] for i in idx])
                s_recs += match_images(same, chosen)
            unrelated.append({m: float(np.mean([r[m] for r in u_recs])) for m in METRICS})
            same_class.append({m: float(np.mean([r[m] for r in s_recs])) for m in METRICS})
        add(f"unrelated digits, best-matched, K={k}", unrelated)
        add(f"same-class digits, best-matched, K={k} (knows labels)", same_class)
        # The line a leak must clear: the better label-aware guess, per seed and metric.
        cls = {m: rows["class average (knows labels)"][m][0] for m in METRICS}
        best = []
        for s in same_class:
            pick = {m: max(s[m], cls[m]) for m in ("ssim", "ssim_zeropad", "psnr")}
            pick["mse"] = min(s["mse"], cls["mse"])
            best.append(pick)
        add(f"label-aware baseline, K={k}", best)

    summary = {}
    for name, vals in rows.items():
        summary[name] = {m: dict(zip(("mean", "ci95"), mean_ci(v))) for m, v in vals.items()}
        summary[name]["seeds"] = len(vals["ssim"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump({"dataset": args.dataset, "images": f"{args.dataset} test {first}-{first + num - 1}", "seeds": args.seeds, "ks": args.ks,
                   "baselines": summary, "per_seed": rows}, f, indent=2)

    def fmt(d):
        return f"{d['mean']:.3f}" if np.isnan(d["ci95"]) else f"{d['mean']:.3f} ± {d['ci95']:.3f}"

    lines = [f"# No-information baselines ({args.dataset} test images {first}-{first + num - 1})", "",
             "A reconstruction must score clearly above the **label-aware baseline** to show a leak, "
             "because the attackers are given the labels. SSIM is the standard form; the old "
             "zero-padded SSIM is shown for comparison. ± is the 95% CI over seeds.", "",
             "| Fake reconstruction | SSIM | Old SSIM (zero-padded) | PSNR (dB) | MSE |",
             "|---|---|---|---|---|"]
    for name, s in summary.items():
        lines.append(f"| {name} | {fmt(s['ssim'])} | {fmt(s['ssim_zeropad'])} | {fmt(s['psnr'])} "
                     f"| {fmt(s['mse'])} |")
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved -> {args.out}.json, {args.out}.md")


if __name__ == "__main__":
    main()
