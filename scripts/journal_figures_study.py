"""Journal figures from the single-step study (saved results only; no attack is rerun)."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import figstyle  # noqa: E402
from src.figstyle import BASE, DOUBLE_COL, INK, INK_2, SETTING_COLORS, SETTING_MARKERS, SETTING_NAMES  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["mnist-fresh", "mnist-trained", "cifar10-fresh", "mnist-fedavg"]
NOISE_ORDER = ["mnist-fresh", "mnist-trained", "cifar10-fresh"]  # FedAvg update has no noise runs
ATTACKER_TITLES = {"basic": "Basic attack (iDLG, L-BFGS)", "cosine": "Cosine attack (Geiping et al.)"}

# Ordinal blue ramp for K (palette sequential steps 250, 400, 550, 700), plus a marker each.
K_COLORS = {1: "#86b6ef", 2: "#3987e5", 4: "#1c5cab", 8: "#0d366b"}
K_MARKERS = {1: "o", 2: "s", 4: "^", 8: "D"}


def _clear(d):
    return d is not None and np.isfinite(d["half"]) and d["lo"] > 0


def _half(d):
    return d["half"] if np.isfinite(d["half"]) else 0.0


def load_study(root="results/study", models_dir="results/models"):
    from scripts.compare_settings import load_all
    loaded, missing = load_all(root, models_dir, ORDER)
    if missing:
        print("journal_figures_study: no results for", ", ".join(missing))
    return loaded


def fig_leak_vs_k(loaded, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.9), sharey=True)
    names = [n for n in ORDER if n in loaded]
    for ax, att in zip(axes, ("basic", "cosine")):
        ax.axhline(0, color=BASE, lw=1.0, ls="--", zorder=1)
        for i, name in enumerate(names):
            _, R = loaded[name]
            pts = [(k, R["grid"][att][(k, 0.0)]) for k in R["ks_leak"] if R["grid"][att].get((k, 0.0))]
            if not pts:
                continue
            c, mk = SETTING_COLORS[name], SETTING_MARKERS[name]
            off = 2 ** ((i - (len(names) - 1) / 2) * 0.05)  # small x offset so bars do not overlap
            x = np.array([k * off for k, _ in pts])
            m = np.array([d["mean"] for _, d in pts])
            ax.plot(x, m, color=c, lw=1.2, zorder=2)
            for xi, (k, d) in zip(x, pts):
                filled = _clear(d)
                ax.errorbar(xi, d["mean"], yerr=_half(d), color=c, lw=0.9, capsize=2, zorder=3,
                            marker=mk, ms=4.5, mfc=c if filled else "white", mec=c, mew=1.0)
            ax.plot([], [], color=c, marker=mk, ms=4.5, label=SETTING_NAMES[name])
        ks = sorted({k for _, R in loaded.values() for k in R["ks_leak"]})
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.minorticks_off()
        ax.set_xlabel("K, examples in the aggregate")
        ax.set_title(ATTACKER_TITLES[att], color=INK)
    axes[0].set_ylabel("Attack SSIM − label-aware baseline")
    # The CIFAR-10 basic K = 1 interval is very wide (one seed failed)
    lo = min(-0.45, axes[0].get_ylim()[0])
    for ax in axes:
        ax.set_ylim(max(lo, -0.6), 1.0)
    d = loaded.get("cifar10-fresh", (None, {"grid": {"basic": {}}}))[1]["grid"]["basic"].get((1, 0.0))
    if d is not None and d["mean"] + _half(d) > 1.0:
        axes[0].annotate(f"interval to {d['mean'] + _half(d):.2f}", xy=(1, 1.0), xytext=(1.25, 0.9),
                         fontsize=6.5, color=INK_2, arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.6))
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles += [Line2D([], [], color=INK_2, marker="o", ls="", mfc=INK_2, ms=4.5),
                Line2D([], [], color=INK_2, marker="o", ls="", mfc="white", ms=4.5)]
    labels += ["clear leak (95% interval > 0)", "no clear leak"]
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    return figstyle.save(fig, out_dir, "fig_study_leak_vs_k")


def _sigma_tick(s):
    """Compact sigma label: 0, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1."""
    if s == 0:
        return "0"
    if s >= 0.01:
        return f"{s:g}"
    m, e = f"{s:.0e}".split("e")
    return f"{m}e{int(e)}"


def _eps_tick(s):
    from scripts.recompute_epsilon import epsilon_replace_one
    if s == 0:
        return "∞"
    e = epsilon_replace_one(s)
    if e < 1000:
        return f"{e:.0f}"
    m, ex = f"{e:.1e}".split("e")
    return f"{m}e{int(ex)}"


def fig_noise(loaded, out_dir):
    names = [n for n in NOISE_ORDER if n in loaded and loaded[n][1]["has_noise"]]
    if not names:
        print("skip fig_study_noise: no noise runs")
        return []
    fig, axes = plt.subplots(2, len(names), figsize=(DOUBLE_COL, 4.9), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for col, name in enumerate(names):
        _, R = loaded[name]
        sigmas = R["sigmas"]
        xpos = {s: i for i, s in enumerate(sigmas)}  # sigma = 0 sits at the left, evenly spaced
        ks = [k for k in R["ks"] if k in K_COLORS]
        for row, att in enumerate(("cosine", "basic")):
            ax = axes[row, col]
            ax.axhline(0, color=BASE, lw=1.0, ls="--", zorder=1)
            for j, k in enumerate(ks):
                pts = [(s, R["grid"][att].get((k, s))) for s in sigmas]
                pts = [(s, d) for s, d in pts if d is not None]
                off = (j - (len(ks) - 1) / 2) * 0.09
                x = np.array([xpos[s] + off for s, _ in pts])
                c, mk = K_COLORS[k], K_MARKERS[k]
                ax.plot(x, [d["mean"] for _, d in pts], color=c, lw=1.1, zorder=2)
                for xi, (s, d) in zip(x, pts):
                    filled = _clear(d)
                    ax.errorbar(xi, d["mean"], yerr=_half(d), color=c, lw=0.8, capsize=1.5, zorder=3,
                                marker=mk, ms=3.8, mfc=c if filled else "white", mec=c, mew=0.9)
                if row == 0 and col == 0:
                    ax.plot([], [], color=c, marker=mk, ms=3.8, label=f"K = {k}")
            ax.set_xticks(range(len(sigmas)))
            if row == 0:
                ax.set_title(SETTING_NAMES[name], color=INK)
            if row == 1:
                ax.set_xticklabels([f"{_sigma_tick(s)}\n{_eps_tick(s)}" for s in sigmas], fontsize=6)
            if col == 0:
                ax.set_ylabel(f"{'Cosine' if att == 'cosine' else 'Basic'} attack\nSSIM − label-aware baseline")
    for ax in axes[1]:
        ax.text(-0.02, -0.062, "σ\nε", transform=ax.transAxes, ha="right", va="top", fontsize=6, color=INK_2)
    axes[0, 0].set_ylim(-0.6, 1.0)
    fig.text(0.5, 0.0, "Noise multiplier σ on the clipped sum (top row of labels); replace-one ε for one release, "
             "δ = 1e-5 (bottom row)", ha="center", va="top", fontsize=7, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles += [Line2D([], [], color=INK_2, marker="o", ls="", mfc=INK_2, ms=3.8),
                Line2D([], [], color=INK_2, marker="o", ls="", mfc="white", ms=3.8)]
    labels += ["clear leak", "no clear leak"]
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    return figstyle.save(fig, out_dir, "fig_study_noise")


DP_LABELS = ["Plain FedAvg (no DP)", "Opacus, C = 10⁶, σ = 0", "C = 1, σ = 0", "C = 1, σ = 1",
             "C = 2, σ = 1", "C = 5, σ = 1", "C = 1, σ = 1, lr 0.03", "C = 1, σ = 1, lr 0.1"]


def fig_dp_utility(out_dir):
    from scripts.plot_dp import ARMS, accs, load
    data = {p: load(p) for p, _ in ARMS}
    missing = [p for p, _ in ARMS if not data[p]]
    if missing:
        print("skip fig_dp_utility: missing", ", ".join(missing))
        return []
    fig, ax = plt.subplots(figsize=(figstyle.SINGLE_COL, 2.6))
    n = len(ARMS)
    for i, (p, _) in enumerate(ARMS):
        y = n - 1 - i
        a = accs(data[p])
        m, sd = a.mean(), a.std(ddof=1)
        ax.scatter(a, np.full(len(a), y) + np.linspace(-0.13, 0.13, len(a)), s=9, color=BASE, zorder=2,
                   linewidths=0)
        ax.errorbar(m, y, xerr=sd, color=figstyle.MULTI, marker="D", ms=4, capsize=2, lw=1.0, zorder=3)
        ax.text(1.0015, y, f"{m:.3f}", va="center", ha="left", fontsize=6.5, color=INK_2, clip_on=False)
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(DP_LABELS)), fontsize=6.5)
    ax.set_xlim(0.895, 1.0)
    ax.set_xlabel("Final test accuracy (MNIST, 3 seeds)")
    ax.grid(axis="y", visible=False)
    ax.axhline(n - 2.5, color=figstyle.GRID, lw=0.8)  # no-DP arms above, DP arms below
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color=BASE, marker="o", ls="", ms=3),
               Line2D([], [], color=figstyle.MULTI, marker="D", ms=4)],
              ["seed", "mean ± sample SD"], loc="center right", bbox_to_anchor=(1.0, 0.5), fontsize=6.5)
    fig.tight_layout()
    return figstyle.save(fig, out_dir, "fig_dp_utility")


def make_all(out_dir="results/journal/figures"):
    figstyle.apply()
    paths = []
    loaded = load_study()
    if loaded:
        paths += fig_leak_vs_k(loaded, out_dir)
        paths += fig_noise(loaded, out_dir)
    paths += fig_dp_utility(out_dir)
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="results/journal/figures")
    args = p.parse_args()
    make_all(args.out)


if __name__ == "__main__":
    main()
