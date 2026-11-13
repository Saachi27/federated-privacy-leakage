"""DP accuracy cost (RQ3): which part of DP-SGD costs accuracy, clipping or noise?"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plotting import MUTED, SERIES, plt, style

# (file prefix, bar label)
ARMS = [
    ("results/runs/train_mnist_baseline", "Plain FedAvg\n(no DP)"),
    ("results/checks/dp_noclip_nonoise", "Opacus on,\nno clip, no noise"),
    ("results/checks/dp_clip1_nonoise", "Clip 1,\nno noise"),
    ("results/runs/train_mnist_dp1", "σ=1, clip 1\n(main DP run)"),
    ("results/checks/dp_sigma1_clip2", "σ=1, clip 2"),
    ("results/checks/dp_sigma1_clip5", "σ=1, clip 5"),
    ("results/checks/dp_sigma1_lr0.03", "σ=1, clip 1,\nlr 0.03"),
    ("results/checks/dp_sigma1_lr0.1", "σ=1, clip 1,\nlr 0.1"),
]


def load(prefix):
    runs = []
    for f in sorted(glob.glob(prefix + "_s*.json")):
        with open(f) as fh:
            d = json.load(fh)
        runs.append({"acc": d["history"][-1]["test_acc"], "privacy": d.get("privacy") or {},
                     "config": d.get("config") or {}})
    return runs


def accs(runs):
    return np.array([r["acc"] for r in runs])


def plot_cost(data, out):
    labels = [label for _, label in ARMS]
    means = [accs(data[p]).mean() for p, _ in ARMS]
    stds = [accs(data[p]).std(ddof=1) for p, _ in ARMS]
    colors = [MUTED, MUTED, SERIES[1], SERIES[0], SERIES[2], SERIES[2], SERIES[3], SERIES[3]]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(ARMS))
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=3, width=0.65)
    for i, (p, _) in enumerate(ARMS):
        a = accs(data[p])
        ax.scatter(np.full(len(a), i) + np.linspace(-0.12, 0.12, len(a)), a, s=12, color="black", zorder=3)
        ax.text(i, means[i] + stds[i] + 0.003, f"{means[i]:.3f}", ha="center", fontsize=8.5)
    base, clip_only, dp = means[0], means[2], means[3]
    # Arrows on the right side: clipping cost and noise cost.
    ax.annotate("", xy=(2.45, clip_only), xytext=(2.45, base),
                arrowprops=dict(arrowstyle="<->", color=SERIES[1], lw=1.4))
    ax.text(2.52, (base + clip_only) / 2, f"clipping\n−{(base - clip_only) * 100:.1f} pts",
            color=SERIES[1], fontsize=9, va="center")
    ax.annotate("", xy=(3.45, dp), xytext=(3.45, clip_only),
                arrowprops=dict(arrowstyle="<->", color=SERIES[0], lw=1.4))
    ax.annotate(f"noise σ=1\n−{(clip_only - dp) * 100:.1f} pts", xy=(3.45, (clip_only + dp) / 2),
                xytext=(3.25, 0.955), color=SERIES[0], fontsize=9, ha="left",
                arrowprops=dict(arrowstyle="-", color=SERIES[0], lw=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0.88, 1.0)
    ax.set_ylabel("Final test accuracy (mean ± std, 3 seeds)")
    ax.set_title("Where DP-SGD loses accuracy (MNIST, 10 clients, 20 rounds; dots = seeds)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_eps(data, out):
    pts = []
    for p, sigma in [("results/runs/train_mnist_dp0.5", 0.5), ("results/runs/train_mnist_dp1", 1.0),
                     ("results/runs/train_mnist_dp2", 2.0)]:
        runs = data[p]
        a = accs(runs)
        eps = np.mean([r["privacy"]["epsilon"] for r in runs])
        pts.append((sigma, eps, a.mean(), a.std(ddof=1)))
    priv = data["results/runs/train_mnist_dp1"][0]["privacy"]
    clip_only = accs(data["results/checks/dp_clip1_nonoise"])
    base = accs(data["results/runs/train_mnist_baseline"])

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    eps = [p[1] for p in pts]
    ax.errorbar(eps, [p[2] for p in pts], yerr=[p[3] for p in pts], marker="o", color=SERIES[0],
                capsize=3, label="DP-SGD, clip 1")
    for sigma, e, m, _ in pts:
        ax.annotate(f"σ={sigma:g}\nε={e:.2f}", (e, m), textcoords="offset points", xytext=(6, -22), fontsize=8.5)
    # Ε = ∞ points drawn at a stand-in position to the right, with a broken-axis note.
    x_inf = 60
    ax.errorbar([x_inf], [clip_only.mean()], yerr=[clip_only.std(ddof=1)], marker="s", color=SERIES[1],
                capsize=3, label="Clip 1, no noise (ε = ∞)")
    ax.errorbar([x_inf], [base.mean()], yerr=[base.std(ddof=1)], marker="D", color=MUTED,
                capsize=3, label="No DP (ε = ∞)")
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 10, 21, x_inf])
    ax.set_xticklabels(["1", "3", "10", "21", "∞"])
    ax.set_xlabel("ε per client (δ = 1e-5)")
    ax.set_ylabel("Final test accuracy (mean ± std, 3 seeds)")
    ax.set_ylim(0.88, 1.0)
    ax.set_title("Accuracy vs privacy budget (MNIST)")
    note = (f"Sampling rate {priv['sample_rate']:.4f} (batch 64 of ~6,000 per client), "
            f"{priv['steps']} steps, clip norm 1, RDP accounting")
    ax.text(0.0, -0.2, note, transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.legend(loc="center left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def write_md(data, out):
    lines = ["# DP Accuracy Cost (RQ3)", "",
             "MNIST, 10 clients, 20 rounds, batch 64, lr 0.01 unless stated. Final test accuracy, "
             "mean ± std over seeds. DP runs report ε per client at δ = 1e-5.", "",
             "| Setting | Accuracy | n | ε |", "|---|---|---|---|"]
    for p, label in ARMS:
        runs = data[p]
        a = accs(runs)
        e = runs[0]["privacy"].get("epsilon") if runs[0]["privacy"] else None
        e_s = "—" if e is None else ("∞" if not np.isfinite(e) else f"{e:.2f}")
        lines.append(f"| {label.replace(chr(10), ' ')} | {a.mean():.4f} ± {a.std(ddof=1):.4f} | {len(a)} | {e_s} |")
    m = {p: accs(data[p]).mean() for p, _ in ARMS}
    base, opac, clip, dp = (m["results/runs/train_mnist_baseline"], m["results/checks/dp_noclip_nonoise"],
                            m["results/checks/dp_clip1_nonoise"], m["results/runs/train_mnist_dp1"])
    lines += ["",
              f"- **Clipping causes almost all of the loss.** Clipping to norm 1 with no noise drops accuracy "
              f"from {base:.4f} to {clip:.4f} ({(base - clip) * 100:.1f} points). Adding σ = 1 noise costs only "
              f"{(clip - dp) * 100:.1f} more points. Opacus itself, with no clip and no noise, costs nothing "
              f"({opac:.4f}).",
              f"- **Tuning wins back about 1–2 points at the same ε.** A larger clip (2 or 5) or a larger "
              f"learning rate (0.03 or 0.1) reaches {m['results/checks/dp_sigma1_clip2']:.3f}–"
              f"{m['results/checks/dp_sigma1_lr0.1']:.3f}. ε depends only on σ, sampling rate and steps, "
              "so these arms keep the ε of σ = 1.", "",
              "Figures: `results/fig_dp_cost.png`, `results/fig_accuracy_vs_eps.png`."]
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    style()
    prefixes = {p for p, _ in ARMS} | {"results/runs/train_mnist_dp0.5", "results/runs/train_mnist_dp2"}
    data = {p: load(p) for p in prefixes}
    missing = [p for p, runs in data.items() if not runs]
    if missing:
        print("Missing runs: " + ", ".join(missing))
        return
    plot_cost(data, "results/fig_dp_cost.png")
    plot_eps(data, "results/fig_accuracy_vs_eps.png")
    write_md(data, "results/dp_check.md")
    print("Saved -> results/fig_dp_cost.png, results/fig_accuracy_vs_eps.png, results/dp_check.md")


if __name__ == "__main__":
    main()
