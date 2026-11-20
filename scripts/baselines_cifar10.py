"""No-information SSIM baselines for the CIFAR-10 attack table."""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.baselines import mean_ci, score
from src import splits
from src.data import load_datasets

NUM = 25  # scripts/evaluate.py --dataset cifar10
SPLITS = {"legacy": splits.LEGACY_CIFAR10_BASELINE, "journal": splits.JOURNAL}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--splits", choices=list(SPLITS), default="legacy",
                   help="image ranges from src/splits.py (legacy = old table)")
    p.add_argument("--num-images", type=int, default=NUM)
    p.add_argument("--out", default="results/baselines_cifar10")
    args = p.parse_args()
    split, num = SPLITS[args.splits], args.num_images

    train_ds, test_ds, _ = load_datasets("cifar10", normalize=False)
    first = split.targets.start
    assert split.targets.contains(first, num, len(test_ds)), \
        f"{num} targets do not fit in the {split.name} target range {split.targets}"
    if args.splits != "legacy":
        assert split.is_disjoint(len(test_ds)), split.overlaps(len(test_ds))
    targets = [test_ds[i][0] for i in range(first, first + num)]
    pool_idx = split.prior.indices(len(test_ds))
    average = torch.tensor(train_ds.data, dtype=torch.float32).mean(0).permute(2, 0, 1) / 255.0

    rows = {}

    def add(name, seed_scores):
        rows[name] = {m: [s[m] for s in seed_scores] for m in ("ssim", "psnr", "mse")}

    add("all black", [score([torch.zeros_like(t) for t in targets], targets)])
    add("mid grey", [score([torch.full_like(t, 0.5) for t in targets], targets)])
    add("average image", [score([average for _ in targets], targets)])
    noise, unrelated = [], []
    for seed in args.seeds:
        g = torch.Generator().manual_seed(seed)
        noise.append(score([torch.rand(t.shape, generator=g) for t in targets], targets))
        idx = np.random.default_rng(1000 + seed).choice(pool_idx, num, replace=False)
        unrelated.append(score([test_ds[int(i)][0] for i in idx], targets))
    add("random noise", noise)
    add("unrelated image", unrelated)

    summary = {name: {m: dict(zip(("mean", "ci95"), mean_ci(v))) for m, v in vals.items()}
               for name, vals in rows.items()}
    with open(args.out + ".json", "w") as f:
        json.dump({"images": f"CIFAR-10 test {first}-{first + num - 1}", "splits": split.name,
                   "seeds": args.seeds,
                   "baselines": summary, "per_seed": rows}, f, indent=2)

    def fmt(d):
        return f"{d['mean']:.3f}" if np.isnan(d["ci95"]) else f"{d['mean']:.3f} ± {d['ci95']:.3f}"

    lines = [f"# No-information baselines, CIFAR-10 (test images {first}-{first + num - 1}, "
             f"{len(args.seeds)} seeds)", "",
             "| Fake reconstruction | SSIM | PSNR | MSE |", "|---|---|---|---|"]
    for name, d in summary.items():
        lines.append(f"| {name} | {fmt(d['ssim'])} | {fmt(d['psnr'])} | {fmt(d['mse'])} |")
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
