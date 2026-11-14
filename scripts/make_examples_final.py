"""Report figure: what the attacker recovers, from one upload to a noisy sum."""

import argparse
import multiprocessing
import os
import sys
import warnings

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.attack_noise_sum import noisy_mean_gradient
from scripts.evaluate import load_images
from src import attack, metrics
from src.evaluation import build_attack_model, default_params, defended_gradient
from src.plotting import plt, style

# (column title, kind, K, sigma)
COLUMNS = [
    ("one upload", "plain", 1, 0.0),
    ("one masked\nupload", "mask", 1, 0.0),
    ("sum, K=2", "sum", 2, 0.0),
    ("sum, K=4", "sum", 4, 0.0),
    ("sum, K=8", "sum", 8, 0.0),
    ("K=2\nσ=1e-4", "sum", 2, 1e-4),
    ("K=2\nσ=1e-3", "sum", 2, 1e-3),
    ("K=2\nσ=1e-2", "sum", 2, 1e-2),
]
ATTACKER_NAMES = {"basic": "Basic (L-BFGS)", "cosine": "Cosine (Geiping)"}


def _init_worker():
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")


def _run(task):
    attacker, idx, col, images, meta, seed, clip = task
    _, kind, k, sigma = COLUMNS[col]
    model = build_attack_model(meta, seed)
    chosen = [images[(idx + j) % len(images)] for j in range(k)]
    xs = [c[0] for c in chosen]
    labels = [c[1] for c in chosen]
    if kind == "sum":
        per_example = [attack.compute_gradient(model, x[None], torch.tensor([y])) for x, y in zip(xs, labels)]
        gen = torch.Generator().manual_seed(seed * 1_000_003 + k * 1009 + idx)
        z = [torch.randn(g.shape, generator=gen) for g in per_example[0]]
        observed, _ = noisy_mean_gradient(per_example, clip, sigma, z=z)
    else:
        grads = attack.compute_gradient(model, xs[0][None], torch.tensor(labels))
        observed = grads if kind == "plain" else defended_gradient(grads, "mask", default_params(def_seed=seed))
    shape = (k,) + tuple(xs[0].shape)
    if attacker == "cosine":
        recon, _ = attack.reconstruct_cosine(model, observed, shape, labels, iters=3000, tv_weight=1e-3, seed=seed)
    else:
        recon, _ = attack.reconstruct(model, observed, shape, meta["num_classes"], label=labels, iters=300,
                                      seed=seed)
    recon = torch.nan_to_num(recon, nan=0.0, posinf=1.0, neginf=0.0)
    scores = np.array([[metrics.all_metrics(recon[i], xs[j])["ssim"] for j in range(k)] for i in range(k)])
    rows, cols = linear_sum_assignment(scores, maximize=True)
    i = int(rows[list(cols).index(0)])  # reconstruction paired with the shown original
    return attacker, idx, col, recon[i].clamp(0, 1), float(scores[i, 0])


def show(ax, img):
    ax.imshow(img.detach().cpu()[0], cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def baseline_note(path):
    """'; label-aware no-information baseline ≈ lo–hi' from the baselines file, or ''."""
    import json
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        base = json.load(f)["baselines"]
    vals = [v["ssim"]["mean"] for name, v in base.items()
            if name.startswith("label-aware baseline") and not name.endswith("K=16")]
    if not vals:
        return ""
    return f"; no-information baseline, knowing labels ≈ {min(vals):.2f}–{max(vals):.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", nargs="+", type=int, default=[0, 5, 10], help="indices into the reported set")
    p.add_argument("--attackers", nargs="+", default=["basic", "cosine"])
    p.add_argument("--first-image", type=int, default=50)
    p.add_argument("--num-images", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clip", type=float, default=30.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="results/fig_examples_final.png")
    p.add_argument("--baselines", default="results/study/baselines_mnist.json")
    p.add_argument("--no-title", action="store_true", help="leave out the in-image title")
    args = p.parse_args()

    images, meta = load_images("mnist", args.first_image + args.num_images)
    images = images[args.first_image:]
    tasks = [(a, i, c, images, meta, args.seed, args.clip)
             for a in args.attackers for i in args.images for c in range(len(COLUMNS))]
    tasks.sort(key=lambda t: -COLUMNS[t[2]][2])  # big K first
    with multiprocessing.Pool(args.workers, initializer=_init_worker) as pool:
        out = {(a, i, c): (img, s) for a, i, c, img, s in pool.imap_unordered(_run, tasks)}

    style()
    rows = [(a, i) for a in args.attackers for i in args.images]
    ncols = len(COLUMNS) + 1
    fig, axes = plt.subplots(len(rows), ncols, figsize=(1.25 * ncols + 1.2, 1.45 * len(rows)), squeeze=False)
    for r, (a, i) in enumerate(rows):
        show(axes[r][0], images[i][0])
        axes[r][0].set_ylabel(f"{ATTACKER_NAMES.get(a, a)}", fontsize=8, rotation=0, ha="right", va="center",
                              labelpad=6)
        if r == 0:
            axes[r][0].set_title("original", fontsize=9)
        for c in range(len(COLUMNS)):
            img, s = out[(a, i, c)]
            show(axes[r][c + 1], img)
            axes[r][c + 1].set_xlabel(f"{s:.2f}", fontsize=8, labelpad=2,
                                      color="#0b0b0b" if s >= 0.9 else "#52514e",
                                      fontweight="bold" if s >= 0.9 else "normal")
            if r == 0:
                axes[r][c + 1].set_title(COLUMNS[c][0], fontsize=9)
    if not args.no_title:
        fig.suptitle("What the attacker recovers (MNIST, fresh LeNet; number = SSIM, bold = success ≥ 0.9"
                     f"{baseline_note(args.baselines)})", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
