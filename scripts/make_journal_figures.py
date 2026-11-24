"""Every journal figure from saved results, with one command."""

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import figstyle as fs  # noqa: E402

fs.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

SETTINGS = ("mnist-fresh", "mnist-trained")
NAME = {"mnist-fresh": "MNIST fresh", "mnist-trained": "MNIST trained"}
ATT = {"unrolled": "attacker 2", "cosine": "attacker 1, cosine", "lbfgs": "attacker 1, basic"}
EXAMPLE_RELEASES = (0, 10, 20)


def load(path):
    with open(path) as f:
        return json.load(f)


def record(records, setting, n, e, split, attacker, release):
    path = os.path.join(records, setting, f"N{n}", f"E{e}_{split}_{attacker}_r{release:02d}.json")
    return load(path)


def cell_records(records, setting, n, e, split, attacker):
    out = {}
    for path in glob.glob(os.path.join(records, setting, f"N{n}", f"E{e}_{split}_{attacker}_r*.json")):
        r = load(path)
        out[r["release"]] = r
    return out


def release_baselines(path, n):
    """Release id -> label-aware baseline mean SSIM for that release."""
    data = load(path)
    block = data["releases"][str(n)]
    return {int(r["release"]): float(r["b"]) for r in block}


def _box(ax, x, y, w, h, text, edge=fs.INK_2, face="white", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                linewidth=0.8, edgecolor=edge, facecolor=face))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, color=fs.INK,
            weight=weight, linespacing=1.2)


def _arrow(ax, a, b, color=fs.INK_2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color=color,
                                 shrinkA=0, shrinkB=0))


def fig_threat_model(out):
    """Drawn at the paper's text width so the box text stays at 7.5 pt on the page."""
    fig, ax = plt.subplots(figsize=(fs.TEXT_WIDTH, 2.6))
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(-0.012, 1.012)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w, h = 0.16, 0.25
    ys = {"multi": 0.70, "single": 0.31}
    cols = [0.0, 0.21, 0.42, 0.63, 0.84]  # left edge of each column
    _box(ax, cols[0], ys["multi"], w, h, "N clients,\n1 image each\n(N × 1)", edge=fs.MULTI, weight="bold")
    _box(ax, cols[0], ys["single"], w, h, "1 client,\nall N images\n(1 × N)", edge=fs.SINGLE, weight="bold")
    _box(ax, cols[1], ys["multi"], w, h, "Each client\nruns E local\nsteps on\nits image")
    _box(ax, cols[1], ys["single"], w, h, "The client\nruns E local\nsteps on all\nN images")
    _box(ax, cols[2], ys["multi"], w, h, "Masked\nuploads;\nmasks cancel\nin the sum")
    _box(ax, cols[2], ys["single"], w, h, "Full update,\nno masking")
    for k in ys:
        for c in (0, 1):
            _arrow(ax, (cols[c] + w, ys[k] + h / 2), (cols[c + 1], ys[k] + h / 2))
    yr = (ys["multi"] + ys["single"]) / 2
    _box(ax, cols[3], yr, w, h, "Server forms\nU: weighted\nsum ÷ NηE", weight="bold")
    _arrow(ax, (cols[2] + w, ys["multi"] + h / 2), (cols[3], yr + h * 0.8))
    _arrow(ax, (cols[2] + w, ys["single"] + h / 2), (cols[3], yr + h * 0.2))
    _box(ax, cols[4], ys["multi"], w, h, "Attacker 1:\nU as a mean\ngradient")
    _box(ax, cols[4], ys["single"], w, h, "Attacker 2:\nunrolls the\nE local steps")
    _arrow(ax, (cols[3] + w, yr + h * 0.8), (cols[4], ys["multi"] + h / 2))
    _arrow(ax, (cols[3] + w, yr + h * 0.2), (cols[4], ys["single"] + h / 2))
    notes = ("Same images and model in both rows; at E = 1 both give the same mean gradient (positive control).",
             "Attackers get the true labels. Each release is scored against the label-aware baseline:",
             "mean SSIM, any image ≥ 0.9, and gallery retrieval.")
    for yt, note in zip((0.17, 0.09, 0.01), notes):
        ax.text(0.5, yt, note, fontsize=7, color=fs.INK_2, ha="center", va="center")
    return fs.save(fig, out, "fig_threat_model")


def fig_main_paired(out, records, baselines, report):
    s, n = "mnist-fresh", 2
    e1 = cell_records(records, s, n, 1, "1x2", "cosine")
    single = cell_records(records, s, n, 5, "1x2", "unrolled")
    multi = cell_records(records, s, n, 5, "2x1", "unrolled")
    base = release_baselines(baselines, n)
    rel = sorted(multi)
    cols = [("E = 1\n(control)", e1, fs.CONTROL), ("E = 5\n1 × 2", single, fs.SINGLE), ("E = 5\n2 × 1", multi, fs.MULTI)]
    fig, ax = plt.subplots(figsize=(fs.SINGLE_COL, 2.7))
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.06, 0.06, size=len(rel))
    vals = [np.array([np.mean(c[r]["ssim"]) for r in rel]) for _, c, _ in cols]
    for i in range(len(rel)):
        ax.plot([1 + jitter[i], 2 + jitter[i]], [vals[1][i], vals[2][i]], color=fs.GRID, lw=0.7, zorder=1)
    for x, ((label, _, color), v) in enumerate(zip(cols, vals)):
        ax.scatter(x + jitter, v, s=12, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        ax.plot([x - 0.22, x + 0.22], [v.mean()] * 2, color=fs.INK, lw=1.6, zorder=4)
        ax.text(x + 0.25, v.mean(), f"{v.mean():.2f}", va="center", ha="left", fontsize=6.6, color=fs.INK)
    b = np.mean([base[r] for r in rel])
    ax.axhline(b, color=fs.BASE, ls="--", lw=0.9, zorder=2)
    ax.text(2.45, b + 0.02, f"baseline {b:.2f}", ha="right", va="bottom", fontsize=6.4, color=fs.INK_2)
    ax.set_xticks(range(3))
    ax.set_xticklabels([c[0] for c in cols])
    ax.set_xlim(-0.45, 2.55)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Mean SSIM of the release")
    gap = report["main"]["gap"]["t90"] if "gap" in report["main"] else None
    if gap:
        ax.set_title(f"Gap 2 × 1 − 1 × 2: {gap['mean']:.3f} [{gap['lo']:.3f}, {gap['hi']:.3f}] (90%), n = {len(rel)}",
                     fontsize=6.8, color=fs.INK)
    ax.grid(axis="x", visible=False)
    return fs.save(fig, out, "fig_main_paired")


MARKERS = {"unrolled": "o", "cosine": "s", "lbfgs": "D"}


def grouped_forest(out, name, groups, xlabel):
    """A forest plot at the paper's text width, with one bold header row per group."""
    n_lines = sum(1 + len(rows) for _, rows in groups)
    fig, ax = plt.subplots(figsize=(fs.TEXT_WIDTH, 0.165 * n_lines + 0.75))
    ax.axvspan(-0.05, 0.05, color=fs.GRID, alpha=0.8, lw=0, zorder=0)
    ax.axvline(0, color=fs.BASE, lw=0.8, zorder=1)
    ticks, labels, header = [], [], []
    los, his = [], []
    y = n_lines - 1
    for title, rows in groups:
        ticks.append(y)
        labels.append(title)
        header.append(True)
        y -= 1
        for r in rows:
            main = r.get("main", False)
            ax.plot([r["lo"], r["hi"]], [y, y], color=fs.INK, lw=1.4 if main else 0.9, zorder=2,
                    ls=(0, (2, 1.5)) if r.get("descriptive") else "-")
            ax.scatter([r["mean"]], [y], marker=r["marker"], s=30 if main else 18,
                       color=fs.INK if main else "white", edgecolor=fs.INK, linewidth=0.9, zorder=3)
            ax.text(1.02, y, r["outcome"] + (" (main)" if main else ""), va="center", ha="left", fontsize=7,
                    color=fs.INK if main else fs.INK_2, weight="bold" if main else "normal",
                    transform=ax.get_yaxis_transform(), clip_on=False)
            ticks.append(y)
            labels.append(r["label"])
            header.append(False)
            los.append(r["lo"])
            his.append(r["hi"])
            y -= 1
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7.5)
    for tick, is_header in zip(ax.get_yticklabels(), header):
        if is_header:
            tick.set_fontweight("bold")
            tick.set_color(fs.INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel)
    ax.text(0, n_lines - 0.45, "±0.05 margin", ha="center", va="bottom", fontsize=6.5, color=fs.INK_2)
    ax.set_ylim(-0.7, n_lines + 0.35)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(-0.2, min(los) - 0.05), max(0.2, max(his) + 0.05))
    fig.tight_layout()
    return fs.save(fig, out, name)


def fig_gap_forest(out, report):
    groups = []
    for (s, n) in [(b[0], b[1]) for b in report["blocks"]]:
        rows = []
        for att in ("unrolled", "cosine", "lbfgs"):
            key = f"{s}|{n}|{att}"
            if key in report["labels"]:
                lab = report["labels"][key]
                g = lab["gap"]["t90"]
                rows.append({"label": ATT[att], "lo": g["lo"], "mean": g["mean"], "hi": g["hi"],
                             "outcome": lab["outcome"], "marker": MARKERS[att],
                             "main": s == "mnist-fresh" and n == 2 and att == "unrolled"})
        groups.append((f"{NAME[s]}, N = {n}", rows))
    return grouped_forest(out, "fig_gap_forest", groups, "SSIM gap, N × 1 − 1 × N (90% paired interval)")


def fig_risk(out, report):
    cells = report["cells"]
    base = report["baselines"]
    rows = []
    for s in SETTINGS:
        for n in (2, 4):
            for split, color in ((f"1x{n}", fs.SINGLE), (f"{n}x1", fs.MULTI)):
                rows.append((s, n, split, color, cells[f"{s}|{n}|5|{split}|unrolled"]))
    measures = [("mean_ssim", "Mean SSIM", "mean_b"), ("any_pass_rate", "Any image SSIM ≥ 0.9", "any_pass_rate"),
                ("hit_rate", "Gallery retrieval hit", "hit_rate")]
    # Drawn at the paper's text width so labels stay at 7-7.5 pt on the page.
    fig, axes = plt.subplots(1, 3, figsize=(fs.TEXT_WIDTH, 2.9), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, (key, title, bkey) in zip(axes, measures):
        for yi, (s, n, split, color, c) in zip(y, rows):
            ax.plot([0, c[key]], [yi, yi], color=color, lw=1.0, alpha=0.45, solid_capstyle="round")
            ax.scatter([c[key]], [yi], color=color, s=20, edgecolor="white", linewidth=0.5, zorder=3)
            ax.text(c[key] + 0.04, yi, f"{c[key]:.2f}", va="center", fontsize=7, color=fs.INK)
            b = base[str(n)][bkey]
            ax.plot([b, b], [yi - 0.38, yi + 0.38], color=fs.BASE, lw=1.2, zorder=2)
        ax.set_title(title, fontsize=7.5)
        ax.set_xlim(0, 1.35)
        ax.set_xticks([0, 0.5, 1])
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{NAME[s]}, N = {n}, {split.replace('x', ' × ')}" for s, n, split, _, _ in rows],
                            fontsize=7.5)
    handles = [plt.Line2D([], [], marker="o", ls="", color=fs.MULTI, label="N × 1 (N clients)"),
               plt.Line2D([], [], marker="o", ls="", color=fs.SINGLE, label="1 × N (one client)"),
               plt.Line2D([], [], color=fs.BASE, lw=1.2, label="label-aware baseline")]
    fig.suptitle("E = 5, attacker 2, 30 releases per row", fontsize=7, color=fs.INK_2)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.6)
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    return fs.save(fig, out, "fig_risk")


def fig_why(out, why):
    fig, ax = plt.subplots(figsize=(fs.SINGLE_COL, 2.6))
    rng = np.random.default_rng(1)
    labels = []
    for i, c in enumerate(why["cells"]):
        for k, (key, color, dx) in enumerate((("cos_multi_g", fs.MULTI, -0.17), ("cos_single_g", fs.SINGLE, 0.17))):
            v = np.array([r[key] for r in c["rows"]])
            ax.scatter(i + dx + rng.uniform(-0.06, 0.06, len(v)), v, s=9, color=color, edgecolor="white",
                       linewidth=0.4, zorder=3)
            ax.plot([i + dx - 0.12, i + dx + 0.12], [np.median(v)] * 2, color=fs.INK, lw=1.4, zorder=4)
        labels.append(f"{NAME[c['setting']].replace('MNIST ', '')}\nN = {c['n']}")
    ax.axhline(1, color=fs.BASE, lw=0.8, ls="--")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("cos(E = 5 release, E = 1 gradient)")
    ax.set_ylim(-0.35, 1.08)
    ax.grid(axis="x", visible=False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=fs.MULTI, label="N × 1"),
               plt.Line2D([], [], marker="o", ls="", color=fs.SINGLE, label="1 × N"),
               plt.Line2D([], [], color=fs.INK, lw=1.4, label="median")]
    ax.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.2))
    ax.set_title("MNIST, 30 releases per cell (post hoc)", fontsize=7, color=fs.INK_2)
    return fs.save(fig, out, "fig_why")


def _recon(records, rec, target):
    d = np.load(os.path.join(records, rec["setting"], f"N{rec['n']}", rec["recon_file"]))
    return d["recon_slots"][rec["target_to_slot"][target]][0]


def fig_examples(out, records, n):
    from scripts.attack_multiclient import load_images
    cols = [("Original", None), ("E = 1", (1, f"1x{n}", "cosine")),
            (f"1 × {n}\natt. 1", (5, f"1x{n}", "cosine")), (f"1 × {n}\natt. 2", (5, f"1x{n}", "unrolled")),
            (f"{n} × 1\natt. 1", (5, f"{n}x1", "cosine")), (f"{n} × 1\natt. 2", (5, f"{n}x1", "unrolled"))]
    nrows = n * len(EXAMPLE_RELEASES)
    fig, axes = plt.subplots(nrows, 2 * len(cols) + 1, figsize=(fs.DOUBLE_COL, 0.66 * nrows + 0.45),
                             gridspec_kw={"width_ratios": [1] * len(cols) + [0.5] + [1] * len(cols),
                                          "wspace": 0.06, "hspace": 0.32})
    for side, s in enumerate(SETTINGS):
        off = side * (len(cols) + 1)
        row = 0
        for rid in EXAMPLE_RELEASES:
            base_rec = record(records, s, n, 1, f"1x{n}", "cosine", rid)
            images, _, _ = load_images("mnist", base_rec["image_indices"])
            for t in range(n):
                for j, (title, spec) in enumerate(cols):
                    ax = axes[row, off + j]
                    if spec is None:
                        img, txt = images[t, 0].numpy(), None
                    else:
                        rec = record(records, s, n, spec[0], spec[1], spec[2], rid)
                        img, val = _recon(records, rec, t), rec["ssim"][t]
                        txt = f"{val:.2f}"
                    ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
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
        axes[0, off + len(cols) // 2].annotate(NAME[s], xy=(0.5, 1), xycoords="axes fraction",
                                              xytext=(0, 22), textcoords="offset points", ha="center",
                                              fontsize=7.5, weight="bold")
    for r in range(nrows):
        axes[r, len(cols)].axis("off")
    return fs.save(fig, out, f"fig_examples_n{n}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", default="results/journal/targets")
    p.add_argument("--report", default="results/journal/report.json")
    p.add_argument("--why", default="results/journal/why_more.json")
    p.add_argument("--baselines", default="results/journal/baselines_mnist.json")
    p.add_argument("--out", default="results/journal/figures")
    p.add_argument("--skip-study", action="store_true", help="skip the redrawn single-step study figures")
    args = p.parse_args()

    report = load(args.report)
    fig_threat_model(args.out)
    fig_main_paired(args.out, args.records, args.baselines, report)
    fig_gap_forest(args.out, report)
    fig_risk(args.out, report)
    fig_why(args.out, load(args.why))
    fig_examples(args.out, args.records, 2)
    fig_examples(args.out, args.records, 4)
    if not args.skip_study:
        from scripts.journal_figures_study import make_all
        make_all(args.out)


if __name__ == "__main__":
    main()
