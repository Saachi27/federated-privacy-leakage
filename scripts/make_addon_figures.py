"""Figures for the section 21 add-ons, from saved results."""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.make_journal_figures as jf  # noqa: E402
from src import figstyle as fs  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

MARKERS = {"unrolled": "o", "cosine": "s", "lbfgs": "D"}


def _interval(part):
    """The 90% gap interval; the bootstrap one (marked descriptive) if no t-interval."""
    if part.get("t90"):
        return part["t90"], False
    return part.get("boot90"), True


def _row_name(r):
    setting = fs.SETTING_NAMES[r["setting"]]
    split = f"{r['multi_split'].replace('x', '×')} − {r['single_split'].replace('x', '×')}"
    extra = ", E = 10" if r["group"] == "21.B" else ""
    return f"{setting}, {split}{extra}, {jf.ATT[r['attacker']]}"


def fig_addons_forest(out, rows):
    rows = [r for r in rows if r["outcome"] != "not run" and _interval(r["gap"])[0]]
    if not rows:
        print("fig_addons_forest: no add-on rows yet, skipped")
        return None
    fig, ax = plt.subplots(figsize=(fs.DOUBLE_COL * 0.66, 0.24 * len(rows) + 0.7))
    ax.axvspan(-0.05, 0.05, color=fs.GRID, alpha=0.8, lw=0, zorder=0)
    ax.axvline(0, color=fs.BASE, lw=0.8, zorder=1)
    y = np.arange(len(rows))[::-1]
    names = []
    for yi, r in zip(y, rows):
        g, descriptive = _interval(r["gap"])
        ax.plot([g["lo"], g["hi"]], [yi, yi], color=fs.INK, lw=0.9, zorder=2,
                ls=(0, (2, 1.5)) if descriptive else "-")
        ax.scatter([g["mean"]], [yi], marker=MARKERS[r["attacker"]], s=16, color="white",
                   edgecolor=fs.INK, linewidth=0.9, zorder=3)
        ax.text(1.02, yi, r["outcome"], va="center", ha="left", fontsize=6.4, color=fs.INK_2,
                transform=ax.get_yaxis_transform(), clip_on=False)
        names.append(_row_name(r))
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.2)
    ax.set_xlabel("SSIM gap, fewer-images-per-client split − more (90% paired interval)")
    ax.text(0, len(rows) - 0.35, "±0.05\nmargin", ha="center", va="bottom", fontsize=6, color=fs.INK_2)
    ax.set_ylim(-0.7, len(rows) + 0.4)
    ax.grid(axis="y", visible=False)
    los = [_interval(r["gap"])[0]["lo"] for r in rows]
    his = [_interval(r["gap"])[0]["hi"] for r in rows]
    ax.set_xlim(min(-0.2, min(los) - 0.05), max(0.2, max(his) + 0.05))
    return fs.save(fig, out, "fig_addons_forest")


def _as_image(arr):
    """A saved reconstruction slot or a dataset image -> H x W (grey) or H x W x 3."""
    arr = np.asarray(arr, dtype=np.float32)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.shape[0] == 1:
        return arr[0]
    return np.transpose(arr, (1, 2, 0))


def fig_examples_cifar10(out, records, n=2):
    from scripts.attack_multiclient import load_images
    s = "cifar10-fresh"
    cols = [("Original", None), ("E = 1", (1, f"1x{n}", "cosine")),
            (f"1 × {n}\natt. 1", (5, f"1x{n}", "cosine")), (f"1 × {n}\natt. 2", (5, f"1x{n}", "unrolled")),
            (f"{n} × 1\natt. 1", (5, f"{n}x1", "cosine")), (f"{n} × 1\natt. 2", (5, f"{n}x1", "unrolled"))]
    try:
        for rid in jf.EXAMPLE_RELEASES:
            for _, spec in cols[1:]:
                jf.record(records, s, n, spec[0], spec[1], spec[2], rid)
    except FileNotFoundError as err:
        print(f"fig_examples_cifar10: missing record ({err.filename}), skipped")
        return None
    nrows = n * len(jf.EXAMPLE_RELEASES)
    fig, axes = plt.subplots(nrows, len(cols), figsize=(fs.SINGLE_COL * 1.15, 0.62 * nrows + 0.45),
                             gridspec_kw={"wspace": 0.06, "hspace": 0.32})
    row = 0
    for rid in jf.EXAMPLE_RELEASES:
        base = jf.record(records, s, n, 1, f"1x{n}", "cosine", rid)
        images, _, _ = load_images("cifar10", base["image_indices"])
        for t in range(n):
            for j, (title, spec) in enumerate(cols):
                ax = axes[row, j]
                if spec is None:
                    img, txt = _as_image(images[t].numpy()), None
                else:
                    rec = jf.record(records, s, n, spec[0], spec[1], spec[2], rid)
                    d = np.load(os.path.join(records, s, f"N{n}", rec["recon_file"]))
                    img = _as_image(d["recon_slots"][rec["target_to_slot"][t]])
                    txt = f"{rec['ssim'][t]:.2f}"
                ax.imshow(np.clip(img, 0, 1), interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.grid(False)
                if txt:
                    ax.set_xlabel(txt, fontsize=6, labelpad=1, color=fs.INK,
                                  weight="bold" if float(txt) >= 0.9 else "normal")
                if row == 0:
                    ax.set_title(title, fontsize=6.4, pad=3)
                if j == 0:
                    ax.set_ylabel(f"r{rid}" if t == 0 else "", fontsize=6, rotation=0, labelpad=8,
                                  va="center", color=fs.INK_2)
            row += 1
    return fs.save(fig, out, "fig_examples_cifar10")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", default="results/journal/addons_report.json")
    p.add_argument("--records", default="results/journal/addons")
    p.add_argument("--out", default="results/journal/figures")
    args = p.parse_args()
    if os.path.exists(args.report):
        fig_addons_forest(args.out, jf.load(args.report))
    else:
        print(f"{args.report} not found: run scripts/analyze_addons.py first")
    fig_examples_cifar10(args.out, args.records)


if __name__ == "__main__":
    main()
