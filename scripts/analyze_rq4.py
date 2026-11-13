"""RQ2 and RQ4: when does noise on the sum stop the attack"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plotting import INK, MUTED, SERIES, MARKERS, plt, style  # noqa: E402

ATTACKERS = [("basic", "Basic (iDLG, L-BFGS)"), ("cosine", "Cosine (Geiping et al.)")]
SUCCESS = 0.9
PSNR_CAP = 60.0  # dB; an exact reconstruction has infinite PSNR
BOUND_EXTRA_SIGMAS = [0.3, 1.0, 2.0]
KAPPA_FLOOR = 1e-3
RESERVED = 82  # test images 0-81 are tuning or target images


def load_records(runs_dir, pattern):
    records = []
    for path in sorted(glob.glob(os.path.join(runs_dir, pattern))):
        with open(path) as f:
            records += json.load(f)["records"]
    return records


def load_baselines(path):
    with open(path) as f:
        return json.load(f)


def label_aware(base, k):
    return f"label-aware baseline, K={k}"


def base_seeds(base, k, metric):
    """{seed: label-aware baseline value at K} for a metric, from the per-seed lists."""
    vals = base["per_seed"][label_aware(base, k)][metric]
    seeds = base["seeds"]
    if len(vals) == 1:
        return {s: vals[0] for s in seeds}
    return dict(zip(seeds, vals))


def base_mean(base, k, metric):
    b = base["baselines"][label_aware(base, k)][metric]
    return b["mean"], (b["ci95"] if b["ci95"] is not None and np.isfinite(b["ci95"]) else 0.0)


def _values(r, metric):
    v = np.asarray(r[metric], float)
    if metric == "psnr":
        v = np.minimum(np.nan_to_num(v, nan=0.0, posinf=PSNR_CAP), PSNR_CAP)
    return v


def seed_means(rows, metric="ssim"):
    by = {}
    for r in rows:
        by.setdefault(r["seed"], []).extend(_values(r, metric).tolist())
    return {s: float(np.mean(v)) for s, v in by.items()}


def ci(values):
    """Mean and 95% t half-width of a list of per-seed values (nan with one seed)."""
    a = np.asarray(values, float)
    if len(a) == 0:
        return None
    half = float(stats.t.ppf(0.975, len(a) - 1) * stats.sem(a)) if len(a) >= 2 else float("nan")
    m = float(a.mean())
    return {"mean": m, "half": half, "lo": m - half, "hi": m + half, "n": len(a)}


def paired(a, b):
    """A, b: {seed: value}."""
    common = sorted(set(a) & set(b))
    return ci([a[s] - b[s] for s in common]) if common else None


def cell(rows, metric="ssim"):
    """Mean over seeds of the per-seed mean, its interval, and the success rate."""
    if not rows:
        return None
    c = ci(list(seed_means(rows, metric).values()))
    ssim = np.concatenate([_values(r, "ssim") for r in rows])
    c.update({"success": float((ssim >= SUCCESS).mean()), "images": len(ssim)})
    return c


def select(records, **kw):
    return [r for r in records if all(r.get(k) == v for k, v in kw.items())]


def clear(d):
    return d is not None and np.isfinite(d["half"]) and d["lo"] > 0


def stopped(d):
    return d is not None and np.isfinite(d["half"]) and d["hi"] <= 0


def fmt(c, digits=3, signed=False):
    if c is None:
        return "—"
    m = f"{c['mean']:+.{digits}f}" if signed else f"{c['mean']:.{digits}f}"
    return m if not np.isfinite(c["half"]) else f"{m} ± {c['half']:.{digits}f}"


def sig_label(s):
    """Short σ label: 0, 3e-5, 1e-4, 0.001, 0.1."""
    if s == 0:
        return "0"
    if s >= 1e-3:
        return f"{s:g}"
    m, e = f"{s:.0e}".split("e")
    return f"{m}e{int(e)}"


def _ssim_many(guess, targets, window_size=11, sigma=1.5):
    """Standard SSIM of one guess against every target (valid window, as src/metrics.py)."""
    from src.metrics import _gaussian_window
    a = guess.expand_as(targets)
    b = targets
    window = _gaussian_window(window_size, sigma, 1, b.device, b.dtype)
    conv = lambda x: F.conv2d(x, window)  # noqa: E731
    mu_a, mu_b = conv(a), conv(b)
    sa = conv(a * a) - mu_a ** 2
    sb = conv(b * b) - mu_b ** 2
    sab = conv(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (sa + sb + c2))
    return m.mean(dim=(1, 2, 3))


def kappa_estimate(path, guesses_per_class=50, targets_per_class=200, seed=0):
    """Label-aware prior success: the best fixed guess per class, weighted by class frequency."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    torch.set_num_threads(1)
    from src.data import load_datasets
    train_ds, test_ds, _ = load_datasets("mnist", normalize=False)
    g = torch.Generator().manual_seed(seed)
    train = train_ds.data.float() / 255.0
    test = test_ds.data[RESERVED:].float() / 255.0
    test_y = test_ds.targets[RESERVED:]
    per_class, weights = {}, {}
    for c in range(10):
        pool = test[test_y == c]
        weights[c] = len(pool) / len(test)
        targets = pool[torch.randperm(len(pool), generator=g)[:targets_per_class]][:, None]
        cls = train[train_ds.targets == c]
        cand = [cls.mean(0)] + [cls[i] for i in torch.randperm(len(cls), generator=g)[:guesses_per_class]]
        rates = [float((_ssim_many(x[None, None], targets) >= SUCCESS).float().mean()) for x in cand]
        per_class[str(c)] = {"kappa": max(rates), "best_is_class_average": (int(np.argmax(rates)) == 0) if max(rates) > 0 else None,
                             "targets": len(targets), "guesses": len(cand), "weight": weights[c]}
    kappa = float(sum(weights[c] * per_class[str(c)]["kappa"] for c in range(10)))
    out = {"kappa_hat": kappa, "threshold": SUCCESS, "per_class": per_class,
           "prior": "label-aware: per class, best of the class average and "
                    f"{guesses_per_class} same-class training images, scored with standard SSIM on "
                    f"up to {targets_per_class} same-class test images (index {RESERVED} and up); "
                    "kappa = class-frequency-weighted average"}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def rero_bound(sigma, kappa):
    """Reconstruction-robustness bound for one full-batch Gaussian release."""
    if sigma <= 0:
        return 1.0
    return float(stats.norm.cdf(stats.norm.ppf(kappa) + 1.0 / sigma))


def leak_grid(noise, base, ks, sigmas, metric, base_metric):
    """{attacker: {(k, sigma): paired attack-minus-baseline interval}}"""
    out = {}
    for a, _ in ATTACKERS:
        out[a] = {}
        for k in ks:
            b = base_seeds(base, k, base_metric)
            for s in sigmas:
                rows = select(noise, k=k, sigma=s, attacker=a)
                if rows:
                    out[a][(k, s)] = paired(seed_means(rows, metric), b)
    return out


def first_stop(cells):
    """Lowest sigma from which this and every larger tested sigma satisfy the rule."""
    stop = None
    for s, ok in reversed(cells):
        if ok:
            stop = s
        else:
            break
    return stop


def stopping(noise, ks, sigmas, grid_ssim, grid_psnr):
    rows = []
    for a, _ in ATTACKERS:
        for k in ks:
            present = [s for s in sigmas if (k, s) in grid_ssim[a]]
            if not present:
                continue
            by_ssim = first_stop([(s, stopped(grid_ssim[a][(k, s)])) for s in present])
            by_psnr = first_stop([(s, stopped(grid_psnr[a].get((k, s)))) for s in present])
            by_succ = first_stop([(s, cell(select(noise, k=k, sigma=s, attacker=a))["success"] <= 0.01)
                                  for s in present])
            rows.append((a, k, by_ssim, by_psnr, by_succ))
    return rows


def decomposition(noise, matched, ks, sigmas, metric):
    rows = []
    for a, _ in ATTACKERS:
        for k in [k for k in ks if k > 1]:
            mix_only = select(noise, k=k, sigma=0.0, attacker=a)
            for s in sigmas:
                ksum = select(noise, k=k, sigma=s, attacker=a)
                informed = select(noise, k=1, sigma=s, attacker=a)
                # With no noise the matched single example is the clean single example.
                msnr = informed if s == 0 else select(matched, k=k, sigma=s, attacker=a)
                if not ksum or not informed:
                    continue
                sm = lambda rows_: seed_means(rows_, metric) if rows_ else {}  # noqa: E731
                rows.append({
                    "attacker": a, "k": k, "sigma": s,
                    "ksum": cell(ksum, metric), "informed": cell(informed, metric),
                    "matched": cell(msnr, metric) if msnr else None,
                    "mix_only": cell(mix_only, metric),
                    "net": paired(sm(ksum), sm(informed)),
                    "benefit": (None if s == 0 else paired(sm(msnr), sm(informed))) if msnr else None,
                    "cost": paired(sm(msnr), sm(ksum)) if msnr else None})
    return rows


def fig_heatmap(grid, ks, sigmas, path, metric_label):
    vals = [d["mean"] for g in grid.values() for d in g.values() if d]
    lo, hi = min(vals + [-0.05]), max(vals + [0.05])
    norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
    for ax, (a, label) in zip(axes, ATTACKERS):
        m = np.full((len(ks), len(sigmas)), np.nan)
        for i, k in enumerate(ks):
            for j, s in enumerate(sigmas):
                d = grid[a].get((k, s))
                if d:
                    m[i, j] = d["mean"]
        im = ax.imshow(m, cmap="RdBu_r", norm=norm, aspect="auto")
        for i, k in enumerate(ks):
            for j, s in enumerate(sigmas):
                d = grid[a].get((k, s))
                if d is None:
                    continue
                txt = f"{d['mean']:+.2f}" + ("*" if clear(d) else "")
                strong = abs(d["mean"]) > 0.6 * max(abs(lo), abs(hi))
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                        color="white" if strong else INK, fontweight="bold" if clear(d) else "normal")
        ax.set_xticks(range(len(sigmas)), [sig_label(s) for s in sigmas])
        ax.set_yticks(range(len(ks)), [str(k) for k in ks])
        ax.set_xlabel("Noise multiplier σ on the sum")
        ax.set_title(label)
        ax.grid(False)
    axes[0].set_ylabel("K, examples in the sum")
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cb.set_label(f"{metric_label} − baseline")
    fig.suptitle("Leak above the label-aware no-information baseline "
                 "(* = 95% interval over seeds above 0)", x=0.45, color=INK)
    fig.savefig(path)
    plt.close(fig)


def fig_decomposition(noise, matched, base, base_metric, sigmas, path, metric, metric_label, ks=(4, 8)):
    ks = [k for k in ks if select(noise, k=k)]
    if not ks:
        return
    fig, axes = plt.subplots(2, len(ks), figsize=(5 * len(ks), 6.4), sharex=True, sharey=True,
                             squeeze=False)
    x = np.arange(len(sigmas))
    for r, (a, label) in enumerate(ATTACKERS):
        for c, k in enumerate(ks):
            ax = axes[r, c]
            bm, bh = base_mean(base, k, base_metric)
            ax.axhspan(bm - bh, bm + bh, color=MUTED, alpha=0.12, lw=0)
            ax.axhline(bm, color=MUTED, ls="--", lw=1.2)
            series = [("Sum of K examples", select(noise, k=k, attacker=a), SERIES[0], MARKERS[0]),
                      ("Informed: knows the other K−1 (same as K=1 in distribution)",
                       select(noise, k=1, attacker=a), SERIES[1], MARKERS[1]),
                      ("Matched SNR: one example, same relative noise (descriptive)",
                       select(matched, k=k, attacker=a), SERIES[2], MARKERS[2])]
            for name, rows, color, marker in series:
                pts = []
                for j, s in enumerate(sigmas):
                    if name.startswith("Matched") and s == 0:
                        cc = cell(select(noise, k=1, sigma=0.0, attacker=a), metric)
                    else:
                        cc = cell([q for q in rows if q["sigma"] == s], metric)
                    if cc:
                        pts.append((j, cc["mean"], cc["half"] if np.isfinite(cc["half"]) else 0))
                if len(pts) <= 1 and name.startswith("Matched"):
                    continue
                if pts:
                    px, py, ph = map(np.array, zip(*pts))
                    ax.plot(px, py, color=color, marker=marker, label=name)
                    ax.fill_between(px, py - ph, py + ph, color=color, alpha=0.15, lw=0)
            ax.set_title(f"{label}, K = {k}")
            if r == 1:
                ax.set_xticks(x, [sig_label(s) for s in sigmas])
                ax.set_xlabel("Noise multiplier σ on the sum")
            if c == 0:
                ax.set_ylabel(f"Mean {metric_label}")
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    handles.append(plt.Line2D([], [], color=MUTED, ls="--"))
    labels.append("Label-aware no-information baseline at that K")
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.07), fontsize=8)
    fig.suptitle("Sum of K vs an attacker who knows the other K−1 vs matched SNR", color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="results/runs")
    p.add_argument("--baselines", default="results/baselines_noise.json")
    p.add_argument("--out-prefix", default="results/rq4")
    args = p.parse_args()
    style()

    noise = load_records(args.runs_dir, "noise_s[0-9]*.json")
    matched = load_records(args.runs_dir, "matched_s[0-9]*.json")
    if not noise:
        sys.exit(f"No {args.runs_dir}/noise_s*.json yet; run scripts/attack_noise_sum.py first.")
    base = load_baselines(args.baselines)
    old = not any("ssim_zeropad" in r for r in noise)
    metric, base_metric = "ssim", ("ssim_zeropad" if old else "ssim")
    metric_label = "old zero-padded SSIM" if old else "SSIM"
    if old:
        print("WARNING: these records only carry the old zero-padded SSIM. Comparing them with the "
              "baselines' zero-padded column; do not report these numbers.")

    ks = sorted({r["k"] for r in noise})
    sigmas = sorted({r["sigma"] for r in noise})
    seeds = sorted({r["seed"] for r in noise})
    mseeds = sorted({r["seed"] for r in matched})
    prelim = len(seeds) < 5 or not matched

    grid = leak_grid(noise, base, ks, sigmas, metric, base_metric)
    grid_psnr = leak_grid(noise, base, ks, sigmas, "psnr", "psnr")
    stops = stopping(noise, ks, sigmas, grid, grid_psnr)
    decomp = decomposition(noise, matched, ks, sigmas, metric)
    kap = kappa_estimate(f"{args.out_prefix}_kappa.json")
    kappa = kap["kappa_hat"] if kap["kappa_hat"] > 0 else KAPPA_FLOOR

    L = ["# RQ2 and RQ4: Noise on the Sum, and What Mixing Protects", ""]
    if old:
        L += ["> **OLD SSIM.** These records were scored with the old zero-padded SSIM, which is "
              "not standard. They are compared with the baselines' zero-padded column. Do not "
              "report these numbers.", ""]
    if prelim:
        L += [f"> **PRELIMINARY.** Noise grid seeds: {seeds or 'none'}. Matched-SNR seeds: "
              f"{mseeds or 'none yet'}. Rerun `python scripts/analyze_rq4.py` when all seeds land.", ""]
    if len(seeds) < 2:
        L += ["> With one seed there are no intervals, so nothing below is marked as a clear "
              "leak or as stopped.", ""]
    L += [f"MNIST test images {base['images'].split()[-1]}, fresh LeNet, clip C = 30, Gaussian "
          "noise σ·C added once to the sum of K clipped per-example gradients, attacker given the "
          "true labels, images scored against their best-matched original. "
          + ("SSIM is the old zero-padded form throughout (see the warning above). " if old
             else "SSIM is the standard form. ")
          + "The leak line is the **label-aware "
          "baseline** at the same K (`results/baselines_noise.md`): the better of the class-average "
          "image and same-class real digits, best-matched, because the attackers know the labels.",
          "",
          "Every ± is a 95% t-interval over seeds of a difference **paired by seed** (the same seed "
          "fixes the model, the grouping of the 16 images and the noise draw).", ""]

    # 1.
    L += ["## 1. Leak Above the Label-Aware Baseline", "",
          "Mean SSIM minus the baseline at the same K, paired by seed. **Bold** = clear "
          "leak (the whole interval is above 0). *Italic* = stopped (the whole interval is at or "
          "below 0). Figure: `results/fig_rq4_heatmap.png`.", ""]
    for a, label in ATTACKERS:
        L += [f"**{label}**", "", "| K | " + " | ".join(f"σ={sig_label(s)}" for s in sigmas) + " |",
              "|---|" + "---|" * len(sigmas)]
        for k in ks:
            cells = []
            for s in sigmas:
                d = grid[a].get((k, s))
                v = fmt(d, signed=True)
                cells.append(f"**{v}**" if clear(d) else f"*{v}*" if stopped(d) else v)
            L.append(f"| {k} | " + " | ".join(cells) + " |")
        L.append("")

    # 2.
    from scripts.attack_noise_sum import epsilon_for
    e = lambda s: "—" if s is None else ("∞" if s == 0 else f"{epsilon_for(s):.3g}")  # noqa: E731
    f = lambda s: ("not stopped" if s is None  # noqa: E731
                   else "0 (stopped with no noise)" if s == 0 else sig_label(s))
    L += ["## 2. Stopping Noise σ*(K)", "",
          "The lowest tested σ from which the attack is stopped at that σ and at every larger "
          "tested σ. Three rules:",
          "- **SSIM rule (main):** the paired 95% interval of attack minus label-aware "
          "baseline is at or below 0.",
          f"- **PSNR rule (cross-check):** the same test with PSNR (capped at {PSNR_CAP:g} dB) "
          "against the label-aware PSNR baseline.",
          "- **Success rule:** at most 1% of images reach SSIM ≥ 0.9. The 0.9 line misses noisy "
          "but readable digits, so this rule stops too early; it is shown for comparison only.", "",
          "| Attacker | K | Baseline SSIM | σ* (SSIM) | ε at σ* | σ* (PSNR) "
          "| ε at σ* | σ* (success) | ε at σ* |",
          "|---|---|---|---|---|---|---|---|---|"]
    for a, k, s1, s2, s3 in stops:
        bm, _ = base_mean(base, k, base_metric)
        L.append(f"| {a} | {k} | {bm:.3f} | {f(s1)} | {e(s1)} | {f(s2)} | {e(s2)} | {f(s3)} | {e(s3)} |")
    L += ["", "ε is for one Gaussian release (δ = 1e-5, sampling rate 1, one step). It is not "
          "the DP-SGD training ε.", ""]

    # 3.
    L += ["## 3. The Sum vs an Attacker Who Knows the Other Examples", "",
          "Three attacks on the same images at the same σ:",
          "- **Sum:** the attack on the noisy sum of K examples.",
          "- **Informed:** an attacker who already knows the other K−1 examples. It subtracts "
          "their gradients and is left with one example plus noise of the same size. For a single "
          "step on a fresh model this is **equal in distribution** to the K = 1 attack at the same "
          "σ (same noise size, not the same noise draw), so the K = 1 results stand in for it.",
          "- **Matched SNR:** one example attacked alone, with its noise shrunk so that its "
          "noise-to-signal ratio equals the K-sum's (σ′ = σ·‖gᵢ‖ / ‖Σⱼ gⱼ‖), from "
          "`scripts/attack_matched_snr.py`. This is a **descriptive control**: it matches the "
          "overall noise-to-signal ratio, not the information about one example, the gradient "
          "geometry or the optimisation difficulty, so it does not isolate a mechanism.", "",
          "**What to expect before looking.** The informed attacker has at least as much "
          "information as the attacker who sees only the sum: it can always compute the sum "
          "itself. So for an optimal attacker the net effect (Sum − Informed) cannot be positive. "
          "A negative net is therefore not a discovery; the useful result is **how large the "
          "gap is** for these two practical attackers, and at which noise levels it closes.", "",
          "Numbers, each paired by seed:",
          "- **Net = Sum − Informed.** How much less one example leaks inside a sum of K than on "
          "its own with the same noise. Negative means the sum leaks less.",
          "- **Matched − Informed.** How much better a single example does when its noise is "
          "shrunk to the sum's noise-to-signal ratio.",
          "- **Matched − Sum.** How much worse the sum does than a single example at that ratio.",
          ""]
    if not matched:
        L += ["> Matched-SNR results are not in yet, so those columns are blank.", ""]
    L += [f"| Attacker | K | σ | Sum | Informed (K=1) | Matched SNR | Sum at σ=0 "
          "| Net (Sum − Informed) | Matched − Informed | Matched − Sum |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for d in decomp:
        net = fmt(d["net"], signed=True)
        if clear(d["net"]) or stopped(d["net"]):
            net = f"**{net}**"
        L.append(f"| {d['attacker']} | {d['k']} | {sig_label(d['sigma'])} | {fmt(d['ksum'])} | "
                 f"{fmt(d['informed'])} | {fmt(d['matched'])} | {fmt(d['mix_only'])} | {net} | "
                 f"{'0 (same by definition)' if d['sigma'] == 0 else fmt(d['benefit'], signed=True)} | "
                 f"{fmt(d['cost'], signed=True)} |")
    L += ["", "**Bold** net = its 95% interval excludes 0. Where the interval includes 0 the data "
          "cannot tell the sum and the informed attacker apart; that is not evidence that they are "
          "equal.", "", "Figure: `results/fig_rq4_decomposition.png` (K = 4 and 8).", ""]

    # 4.
    L += ["## 4. Measured Success vs an Indicative Reconstruction Bound", "",
          "Reconstruction-robustness (ReRo) bound for one release of the Gaussian mechanism, in "
          "the closed form of Kaissis et al. 2023 (arXiv:2307.03928), which agrees with Hayes "
          "et al. 2023 (arXiv:2302.07225, Corollary 4) at sampling rate 1 and one step:", "",
          "  success ≤ Φ(Φ⁻¹(κ) + 1/σ) = 1 − Φ(Φ⁻¹(1 − κ) − 1/σ)", "",
          "**This bound is indicative, not certified.**",
          f"- κ is the best success an attacker can reach **without** the gradient. The attackers "
          f"know the labels, so κ is label-aware: for each class, the best single fixed guess "
          f"among the class average and {kap['per_class']['0']['guesses'] - 1} "
          f"same-class training images, scored on same-class test digits at standard SSIM ≥ "
          f"{SUCCESS}, then weighted by class frequency. κ̂ = {kap['kappa_hat']:.4g}."
          + (f" That is 0, so we use a floor of κ = {KAPPA_FLOOR:g}." if kap["kappa_hat"] <= 0 else ""),
          "- κ̂ is a **lower estimate** of the true κ: it searches finitely many candidate guesses, "
          "and a better guess may exist. A larger κ gives a larger (weaker) bound, so the true "
          "bound may be weaker than shown. The sampling error of κ̂ is not bounded either.",
          "- The bound is about any attacker, including the informed one, so it covers every K. "
          "Our two attackers are far from optimal, so measured success below the bound is "
          "expected.", "",
          "| σ | ε (one release) | Indicative bound | Basic, K=1 | Cosine, K=1 | Highest over K and attackers |",
          "|---|---|---|---|---|---|"]
    for s in sorted(set(sigmas) | set(BOUND_EXTRA_SIGMAS)):
        meas = {}
        for a, _ in ATTACKERS:
            c = cell(select(noise, k=1, sigma=s, attacker=a))
            meas[a] = "—" if c is None else f"{c['success'] * 100:.0f}%"
        alls = [cell(select(noise, k=k, sigma=s, attacker=a)) for k in ks for a, _ in ATTACKERS]
        alls = [c["success"] for c in alls if c]
        hi = "not tested" if not alls else f"{max(alls) * 100:.0f}%"
        eps = "∞" if s == 0 else f"{epsilon_for(s):.3g}"
        L.append(f"| {sig_label(s)} | {eps} | {rero_bound(s, kappa) * 100:.1f}% | {meas['basic']} | "
                 f"{meas['cosine']} | {hi} |")
    all_sig = sorted(set(sigmas) | set(BOUND_EXTRA_SIGMAS))
    useful = [s for s in all_sig if s > 0 and rero_bound(s, kappa) < 0.99]
    stop_all = [s for s in sigmas if s > 0 and all(
        stopped(grid[a].get((k, s))) for k in ks for a, _ in ATTACKERS)]
    L += [""]
    if useful:
        L.append(f"The indicative bound first drops below 99% at σ = {sig_label(useful[0])} "
                 f"(ε = {epsilon_for(useful[0]):.3g}).")
    if stop_all:
        L.append(f"By the SSIM rule, both attacks are stopped at every K from σ = "
                 f"{sig_label(stop_all[0])} (ε = {epsilon_for(stop_all[0]):.3g}).")
    else:
        L.append("By the SSIM rule, no tested σ stops both attacks at every K.")
    L += [""]

    default = args.out_prefix == "results/rq4"
    heat = "results/fig_rq4_heatmap.png" if default else f"{args.out_prefix}_heatmap.png"
    dec = "results/fig_rq4_decomposition.png" if default else f"{args.out_prefix}_decomposition.png"
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    with open(f"{args.out_prefix}_summary.md", "w") as fh:
        fh.write("\n".join(L) + "\n")
    fig_heatmap(grid, ks, sigmas, heat, metric_label)
    fig_decomposition(noise, matched, base, base_metric, sigmas, dec, metric, metric_label)
    print("\n".join(L))
    print(f"Saved {args.out_prefix}_summary.md, {heat}, {dec}")


if __name__ == "__main__":
    main()
