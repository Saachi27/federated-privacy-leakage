"""Study A example reconstructions (saved results only; no attack is rerun)."""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import figstyle as fs, metrics  # noqa: E402
from src.data import load_datasets  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

SEED, K, GROUP, STOP_SIGMA = 0, 8, 0, 0.01
SETTINGS = [("mnist-fresh", "MNIST fresh"), ("mnist-trained", "MNIST trained")]
ROWS = [("Original", None, None), ("Basic,\nno noise", "basic", 0.0), ("Cosine,\nno noise", "cosine", 0.0),
        (f"Cosine,\nσ = {STOP_SIGMA:g}", "cosine", STOP_SIGMA)]


def load_cell(root, setting):
    """The seed-0 noise records of one setting, the saved images, and the group's originals."""
    with open(os.path.join(root, setting, f"noise_s{SEED}.json")) as f:
        data = json.load(f)
    recon = np.load(os.path.join(root, setting, f"noise_s{SEED}_recon.npz"))
    a = data["args"]
    order = np.random.default_rng(SEED).permutation(a["num_images"])  # attack_noise_sum.make_tasks
    idx = [a["first_image"] + int(i) for i in order[GROUP * K:(GROUP + 1) * K]]
    _, test_ds, _ = load_datasets(a["dataset"], normalize=False)
    originals = torch.stack([test_ds[i][0] for i in idx])
    return data["records"], recon, originals, idx


def find(records, attacker, sigma):
    hits = [r for r in records if r["k"] == K and r["seed"] == SEED and r["group"] == GROUP
            and r["attacker"] == attacker and abs(r["sigma"] - sigma) < 1e-12]
    assert len(hits) == 1, (attacker, sigma, len(hits))
    return hits[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="results/study")
    p.add_argument("--out", default="results/journal/figures")
    args = p.parse_args()
    # One spacer row between the two settings, so each block's title has room.
    nrows = len(SETTINGS) * len(ROWS) + len(SETTINGS) - 1
    ratios = ([1] * len(ROWS) + [0.45]) * (len(SETTINGS) - 1) + [1] * len(ROWS)
    fig, axes = plt.subplots(nrows, K, figsize=(fs.DOUBLE_COL * 0.62, 0.62 * nrows + 0.2),
                             gridspec_kw={"wspace": 0.06, "hspace": 0.36, "height_ratios": ratios})
    for b in range(len(SETTINGS) - 1):
        for ax in axes[(b + 1) * len(ROWS) + b]:
            ax.axis("off")
    for b, (setting, name) in enumerate(SETTINGS):
        records, recon, originals, idx = load_cell(args.root, setting)
        for i, (title, attacker, sigma) in enumerate(ROWS):
            row = b * (len(ROWS) + 1) + i
            rec = None if attacker is None else find(records, attacker, sigma)
            images = originals if rec is None else torch.as_tensor(
                recon[f"r{rec['recon_id']}"].astype(np.float32))
            for j in range(K):
                ax = axes[row, j]
                img = images[j, 0].numpy()
                ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if rec is not None:
                    val = float(metrics.ssim(images[j].clamp(0, 1), originals[j]))
                    assert abs(val - rec["ssim"][j]) < 2e-3, (setting, attacker, sigma, j, val, rec["ssim"][j])
                    ax.set_xlabel(f"{rec['ssim'][j]:.2f}", fontsize=6, labelpad=1, color=fs.INK,
                                  weight="bold" if rec["ssim"][j] >= 0.9 else "normal")
                if j == 0:
                    ax.set_ylabel(title, fontsize=6, rotation=0, ha="right", va="center", labelpad=4,
                                  color=fs.INK_2)
        top = axes[b * (len(ROWS) + 1), 0]
        top.annotate(name, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                     fontsize=7.5, weight="bold", ha="left", va="bottom")
        print(f"{setting}: test images {idx}")
    return fs.save(fig, args.out, "fig_study_examples")


if __name__ == "__main__":
    main()
