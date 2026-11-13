"""Report figures for the attack on the sum, drawn with the no-information baselines."""

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plotting import GRID, INK, MARKERS, MUTED, SERIES, plt, style  # noqa: E402

RES = "results"
RUNS = os.path.join(RES, "runs")  # set in main()
OUT = RES  # set in main()
STUDY = False  # True for the results/study layout
TITLE = "MNIST, fresh LeNet"
SETTING_TITLES = {
    "mnist-fresh": "MNIST, fresh LeNet",
    "mnist-trained": "MNIST, trained LeNet",
    "cifar10-fresh": "CIFAR-10, fresh LeNet",
    "mnist-fedavg": "MNIST, fresh LeNet, FedAvg update",
}
ATTACKERS = [("basic", "Basic (iDLG, L-BFGS)"), ("cosine", "Cosine (Geiping et al.)")]
BLACK, CLASS_AVG = "all black", "class average (knows labels)"


def load_records(pattern):
    records = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            records += json.load(f)["records"]
    return records


def load_baselines(path):
    if not os.path.exists(path):
        print(f"No {path}; run scripts/baselines.py first. Drawing without baselines.")
        return {}
    with open(path) as f:
        return json.load(f)["baselines"]


def score_key(recs):
    """Baseline column to compare with: old records only carry the zero-padded SSIM."""
    if recs and not any("ssim_zeropad" in r for r in recs):
        print("WARNING: these records only carry the old zero-padded SSIM; using the baselines' "
              "zero-padded column. Do not report these figures.")
        return "ssim_zeropad"
    return "ssim"


def y_label(key, what="Mean SSIM", note="95% CI over seeds"):
    form = "old zero-padded form" if key == "ssim_zeropad" else "standard form"
    return f"{what}, {form}\n({note})"


def label_aware(base, k, key="ssim"):
    b = base.get(f"label-aware baseline, K={k}")
    if not b:
        return np.nan, np.nan
    h = b[key]["ci95"]
    return b[key]["mean"], (h if h is not None and np.isfinite(h) else 0.0)


def seed_mean_ci(rows):
    """Mean of per-seed mean SSIM, and its 95% CI half-width over seeds."""
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).extend(r["ssim"])
    means = np.array([np.mean(v) for v in by_seed.values()])
    if len(means) < 2:
        return float(means.mean()), 0.0
    return float(means.mean()), float(stats.t.ppf(0.975, len(means) - 1) * stats.sem(means))


def draw_flat_baselines(ax, base, key="ssim"):
    for name, ls in ((BLACK, ":"), (CLASS_AVG, "-.")):
        if name in base:
            v = base[name][key]["mean"]
            ax.axhline(v, color=MUTED, ls=ls, lw=1.2, label=f"Reference: {name} ({v:.2f})")


def sweep_records():
    """Records of the attack on the sum without noise, one list over every K."""
    if not STUDY:
        return load_records(os.path.join(RUNS, "sum_s[0-9]*.json"))
    noise = [r for r in load_records(os.path.join(RUNS, "noise_s[0-9]*.json")) if r.get("sigma", 0) == 0]
    return noise + load_records(os.path.join(RUNS, "sum16_s[0-9]*.json"))


def fig_ssim_vs_k(base):
    recs = sweep_records()
    if not recs:
        print(f"skip fig_ssim_vs_k: no attack-on-the-sum results in {RUNS}")
        return
    key = score_key(recs)
    longs = load_records(os.path.join(RUNS, "sum_long_s[0-9]*.json"))
    ks = sorted({r["k"] for r in recs})
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, (att, label) in enumerate(ATTACKERS):
        pts = [seed_mean_ci([r for r in recs if r["k"] == k and r["attacker"] == att]) for k in ks]
        m, h = np.array(pts).T
        ax.plot(ks, m, marker=MARKERS[i], color=SERIES[i], label=label)
        ax.fill_between(ks, m - h, m + h, color=SERIES[i], alpha=0.15, lw=0)
        lk = sorted({r["k"] for r in longs if r["attacker"] == att})
        if lk:
            lp = np.array([seed_mean_ci([r for r in longs if r["k"] == k and r["attacker"] == att])
                           for k in lk]).T
            ax.errorbar(lk, lp[0], yerr=lp[1], fmt=MARKERS[i], mfc="white", mec=SERIES[i],
                        ecolor=SERIES[i], ms=8, capsize=3, label=f"{label.split(' ')[0]}, 4x steps")
    um, uh = np.array([label_aware(base, k, key) for k in ks]).T
    if not np.all(np.isnan(um)):
        ax.plot(ks, um, color=INK, ls="--", lw=1.4, label="No information, knows labels (leak line)")
        ax.fill_between(ks, um - uh, um + uh, color=INK, alpha=0.08, lw=0)
    draw_flat_baselines(ax, base, key)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("K, examples summed into one update")
    ax.set_ylabel(y_label(key))
    ax.set_title(f"Attack on the sum vs no-information baselines ({TITLE})")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    out = os.path.join(OUT, "fig_ssim_vs_k.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")


def fig_distributions(base):
    recs = sweep_records()
    if not recs:
        print(f"skip fig_ssim_distributions: no attack-on-the-sum results in {RUNS}")
        return
    key = score_key(recs)
    ks = sorted({r["k"] for r in recs})
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    width = 0.36
    for i, (att, label) in enumerate(ATTACKERS):
        data = [np.concatenate([r["ssim"] for r in recs if r["k"] == k and r["attacker"] == att])
                for k in ks]
        pos = np.arange(len(ks)) + (i - 0.5) * (width + 0.04)
        parts = ax.violinplot(data, positions=pos, widths=width, showmedians=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(SERIES[i])
            body.set_edgecolor(SERIES[i])
            body.set_alpha(0.45)
        parts["cmedians"].set_color(INK)
        ax.plot([], [], color=SERIES[i], lw=6, alpha=0.6, label=label)
    for j, k in enumerate(ks):
        m, _ = label_aware(base, k, key)
        if not np.isnan(m):
            ax.hlines(m, j - 0.45, j + 0.45, color=INK, ls="--", lw=1.4,
                      label="No information, knows labels (leak line)" if j == 0 else None)
    if BLACK in base:
        v = base[BLACK][key]["mean"]
        ax.axhline(v, color=MUTED, ls=":", lw=1.2, label=f"Reference: all-black image ({v:.2f})")
    ax.axhline(0.9, color=GRID, lw=1, zorder=0)
    ax.text(len(ks) - 0.5, 0.905, "success threshold 0.9", ha="right", va="bottom", fontsize=7, color=MUTED)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_ylim(-0.05, 1.05)
    n = {len(np.concatenate([r["ssim"] for r in recs if r["k"] == k and r["attacker"] == att]))
         for k in ks for att, _ in ATTACKERS if any(r["k"] == k and r["attacker"] == att for r in recs)}
    per = f"{n.pop()} images per violin" if len(n) == 1 else "images per violin vary"
    ax.set_ylabel(y_label(key, "SSIM per image", "median line"))
    ax.set_title(f"Per-image SSIM, attack on the sum, {TITLE} ({per})")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    out = os.path.join(OUT, "fig_ssim_distributions.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")


def _noise_level(r):
    for key in ("sigma", "noise_multiplier", "noise", "std"):
        if key in r:
            return float(r[key])
    raise KeyError(f"no noise level key in record: {sorted(r)}")


def fig_ssim_vs_noise(base):
    recs = load_records(os.path.join(RUNS, "noise_s[0-9]*.json"))
    if not recs:
        print(f"skip fig_ssim_vs_noise: no {RUNS}/noise_s*.json yet (scripts/attack_noise_sum.py)")
        return
    if not any(_noise_level(r) > 0 for r in recs):
        print(f"skip fig_ssim_vs_noise: {RUNS} has no noise levels above 0")
        return
    key = score_key(recs)
    attackers = [a for a in ATTACKERS if any(r.get("attacker", "basic") == a[0] for r in recs)]
    ks = sorted({r["k"] for r in recs})
    fig, axes = plt.subplots(1, len(attackers), figsize=(5.2 * len(attackers), 4.0), squeeze=False,
                             sharey=True)
    for ax, (att, label) in zip(axes[0], attackers):
        for i, k in enumerate(ks):
            rows = [r for r in recs if r["k"] == k and r.get("attacker", "basic") == att]
            sig = sorted({_noise_level(r) for r in rows})
            if not sig:
                continue
            m, h = np.array([seed_mean_ci([r for r in rows if _noise_level(r) == s]) for s in sig]).T
            # Sigma = 0 cannot sit on a log axis; draw it at a small stand-in value.
            x = np.array(sig)
            pos = x[x > 0]
            zero_x = (pos.min() / 4) if len(pos) else 1e-4
            x = np.where(x > 0, x, zero_x)
            c = SERIES[i % len(SERIES)]
            ax.plot(x, m, marker=MARKERS[i % len(MARKERS)], color=c, label=f"K={k}")
            ax.fill_between(x, m - h, m + h, color=c, alpha=0.12, lw=0)
            um, _ = label_aware(base, k, key)
            if not np.isnan(um):
                ax.axhline(um, color=c, ls="--", lw=0.9, alpha=0.8)
        if any(_noise_level(r) == 0 for r in recs):
            ax.text(0.01, 0.01, "leftmost point = no noise", transform=ax.transAxes, fontsize=7, color=MUTED)
        draw_flat_baselines(ax, base, key)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Noise σ on the sum")
        ax.set_title(label)
    axes[0][0].set_ylabel(y_label(key))
    axes[0][-1].plot([], [], color=MUTED, ls="--", lw=0.9,
                     label="No information, knows labels, at that K (dashed, same colour)")
    axes[0][-1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle(f"Attack on the noisy sum vs no-information baselines ({TITLE})", fontsize=11)
    out = os.path.join(OUT, "fig_ssim_vs_noise.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")


def fig_pipeline():
    steps = [
        ("Client\ndata", "private\nimages"), ("Local\ntraining", "one FedSGD\nstep"),
        ("Gradient", "per client"), ("Masking", "pairwise masks\ncancel in\nthe sum"),
        ("Sum\nover K", "server sees\nonly the sum"), ("Noise", "clip +\nGaussian\n(DP, optional)"),
        ("Server", "honest-but-\ncurious"), ("Attack", "basic or cosine\ninversion"),
        ("Scoring", "SSIM vs\nlabel-aware\nbaseline"),
    ]
    zone = {0: "client", 1: "client", 2: "client", 3: "client", 4: "aggregation", 5: "aggregation",
            6: "server", 7: "server", 8: "evaluation"}
    colors = {"client": SERIES[0], "aggregation": SERIES[2], "server": SERIES[1], "evaluation": SERIES[3]}
    fig, ax = plt.subplots(figsize=(12.5, 3.2))
    ax.set_axis_off()
    ax.grid(False)
    w, gap = 1.1, 0.3
    for i, (title, sub) in enumerate(steps):
        x = i * (w + gap)
        c = colors[zone[i]]
        ax.add_patch(plt.Rectangle((x, 0), w, 1, facecolor=c, alpha=0.14, edgecolor=c, lw=1.4))
        ax.text(x + w / 2, 0.74, title, ha="center", va="center", fontsize=9, weight="bold", color=INK)
        ax.text(x + w / 2, 0.3, sub, ha="center", va="center", fontsize=7, color=MUTED)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap, 0.5), xytext=(x + w, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    for z, c in colors.items():
        ax.plot([], [], color=c, lw=8, alpha=0.4, label=z)
    total = len(steps) * (w + gap) - gap
    ax.set_xlim(-0.05, total + 0.05)
    ax.set_ylim(-0.45, 1.25)
    ax.text(total / 2, 1.12, "Threat model: the server follows the protocol, sees the (noisy) sum, "
            "and is given the true labels", ha="center", fontsize=8.5, color=INK)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.08), fontsize=8)
    out = os.path.join(OUT, "fig_pipeline.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")


def main():
    global RUNS, OUT, STUDY, TITLE
    p = argparse.ArgumentParser()
    p.add_argument("--setting", default=None, help="draw one setting of the study into <root>/<setting>/")
    p.add_argument("--root", default="results/study")
    p.add_argument("--runs-dir", default=None, help="legacy layout (results/runs style)")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    style()
    if args.runs_dir:
        RUNS, OUT, STUDY, TITLE = args.runs_dir, args.out_dir or RES, False, "MNIST, fresh LeNet"
        sweep_base = load_baselines(os.path.join(RES, "baselines.json"))
        noise_base = load_baselines(os.path.join(RES, "baselines_noise.json"))
        draw_pipeline = True
    else:
        from src.settings import SETTINGS
        setting = args.setting or "mnist-fresh"
        RUNS, STUDY, TITLE = os.path.join(args.root, setting), True, SETTING_TITLES.get(setting, setting)
        OUT = args.out_dir or (RES if args.setting is None else RUNS)
        sweep_base = noise_base = load_baselines(
            os.path.join(args.root, f"baselines_{SETTINGS[setting]['dataset']}.json"))
        draw_pipeline = args.setting is None
    os.makedirs(OUT, exist_ok=True)
    fig_ssim_vs_k(sweep_base)
    fig_distributions(sweep_base)
    fig_ssim_vs_noise(noise_base)
    if draw_pipeline:
        fig_pipeline()


if __name__ == "__main__":
    main()
