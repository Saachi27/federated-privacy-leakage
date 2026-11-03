"""Figure for the report: the same images reconstructed under rising protection."""

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_datasets
from src.evaluation import attack_image, default_params
from src.plotting import plt, style

# (column title, defense, DP noise multiplier)
COLUMNS = [
    ("no defense", "none", 0.0), ("clip only", "dp", 0.0), ("σ=0.0001", "dp", 1e-4),
    ("σ=0.001", "dp", 1e-3), ("σ=0.01", "dp", 1e-2), ("σ=1", "dp", 1.0),
    ("mask", "mask", 0.0),
]


def show(ax, img):
    arr = img.detach().cpu()
    if arr.shape[0] == 1:
        ax.imshow(arr[0], cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(arr.permute(1, 2, 0).clamp(0, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p.add_argument("--indices", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--attacker", default="adaptive")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="results/reconstruction_gallery.png")
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    _, test_ds, meta = load_datasets(args.dataset, normalize=False)
    style()
    rows, cols = len(args.indices), len(COLUMNS) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(1.15 * cols, 1.3 * rows), squeeze=False)
    for r, index in enumerate(args.indices):
        image, label = test_ds[index]
        show(axes[r][0], image)
        if r == 0:
            axes[r][0].set_title("original", fontsize=9)
        for c, (title, defense, sigma) in enumerate(COLUMNS, start=1):
            params = default_params(attacker=args.attacker, dp_noise=sigma, model_seed=args.seed,
                                    attack_seed=args.seed, def_seed=args.seed)
            recon, m = attack_image(image, int(label), meta, defense, params)
            show(axes[r][c], recon)
            axes[r][c].set_xlabel(f"{m['ssim']:.2f}", fontsize=8, labelpad=2)
            if r == 0:
                axes[r][c].set_title(title, fontsize=9)
    fig.suptitle(f"What the attacker recovers ({args.attacker} attacker; number = SSIM)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
