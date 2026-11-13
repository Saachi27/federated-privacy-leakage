"""RQ4: as K grows, does the sum's stronger signal or the mixing of K examples win?"""

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

RES = "results"
ATTACKERS = [("basic", "Basic (iDLG, L-BFGS)"), ("cosine", "Cosine (Geiping et al.)")]
SUCCESS = 0.9
BOUND_EXTRA_SIGMAS = [0.3, 1.0, 2.0]
KAPPA_FLOOR = 1e-3


def load_records(pattern):
    records = []
    for path in sorted(glob.glob(os.path.join(RES, "runs", pattern))):
        with open(path) as f:
            records += json.load(f)["records"]
    return records


def load_baselines():
    with open(os.path.join(RES, "baselines.json")) as f:
        return json.load(f)["baselines"]


def base_for(base, k):
    b = base[f"unrelated digits, best-matched, K={k}"]["ssim"]
    return b["mean"], (b["ci95"] if np.isfinite(b["ci95"]) else 0.0)


def cell(rows):
    """Mean SSIM, 95% CI half-width, success rate, number of seeds and images."""
    if not rows:
        return None
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).extend(r["ssim"])
    seed_means = np.array([np.mean(v) for v in by_seed.values()])
    ssim = np.concatenate([np.asarray(r["ssim"], float) for r in rows])
    if len(seed_means) >= 2:
        half = stats.t.ppf(0.975, len(seed_means) - 1) * stats.sem(seed_means)
        over = "seeds"
    else:  # one seed: interval over sums (too narrow, flagged)
        sums = np.array([np.mean(r["ssim"]) for r in rows])
        half = stats.t.ppf(0.975, len(sums) - 1) * stats.sem(sums) if len(sums) > 1 else np.nan
        over = "sums"
    return {"mean": float(seed_means.mean()), "half": float(half), "over": over,
            "success": float((ssim >= SUCCESS).mean()), "seeds": len(seed_means), "images": len(ssim)}


def select(records, **kw):
    return [r for r in records if all(r.get(k) == v for k, v in kw.items())]


def fmt(c, digits=3):
    if c is None:
        return "—"
    half = "n/a" if not np.isfinite(c["half"]) else f"{c['half']:.{digits}f}"
    return f"{c['mean']:.{digits}f} ± {half}"


def sig_label(s):
    """Short σ label: 0, 3e-5, 1e-4, 0.001, 0.1."""
    if s == 0:
        return "0"
    if s >= 1e-3:
        return f"{s:g}"
    m, e = f"{s:.0e}".split("e")
    return f"{m}e{int(e)}"


def _ssim_many(guess, targets, window_size=11, sigma=1.5):
    """SSIM of one guess against every target, same formula as src/metrics.py."""
    from src.metrics import _gaussian_window
    a = guess.expand_as(targets)
    b = targets
    window = _gaussian_window(window_size, sigma, 1, b.device, b.dtype)
    pad = window_size // 2
    conv = lambda x: F.conv2d(x, window, padding=pad)  # noqa: E731
    mu_a, mu_b = conv(a), conv(b)
    sa = conv(a * a) - mu_a ** 2
    sb = conv(b * b) - mu_b ** 2
    sab = conv(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (sa + sb + c2))
    return m.mean(dim=(1, 2, 3))


def kappa_estimate(n_targets=1000, n_guesses=300, seed=0):
    """Best fixed guess's chance of SSIM >= 0.9 on a random MNIST test digit."""
    path = os.path.join(RES, "rq4_kappa.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    torch.set_num_threads(1)
    from src.data import load_datasets
    train_ds, test_ds, _ = load_datasets("mnist", normalize=False)
    g = torch.Generator().manual_seed(seed)
    ti = torch.randperm(len(test_ds.data), generator=g)[:n_targets]
    gi = torch.randperm(len(train_ds.data), generator=g)[:n_guesses]
    targets = (test_ds.data[ti].float() / 255.0)[:, None]
    train = train_ds.data.float() / 255.0
    guesses = [train[i][None, None] for i in gi]
    guesses += [train.mean(0)[None, None], torch.zeros(1, 1, 28, 28)]
    rates = [float((_ssim_many(x, targets) >= SUCCESS).float().mean()) for x in guesses]
    out = {"kappa_hat": max(rates), "mean_rate": float(np.mean(rates)), "targets": n_targets,
           "guesses": len(guesses), "threshold": SUCCESS,
           "prior": "uniform over MNIST test digits; guesses: random training digits, average, black"}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def rero_bound(sigma, kappa):
    """ReRo success bound for one full-batch Gaussian release (noise multiplier sigma)."""
    if sigma <= 0:
        return 1.0
    return float(stats.norm.cdf(stats.norm.ppf(kappa) + 1.0 / sigma))


def leak_grid(noise, base, ks, sigmas):
    """{attacker: {(k, sigma): dict(diff, comb_half, clear, cell)}}"""
    out = {}
    for a, _ in ATTACKERS:
        out[a] = {}
        for k in ks:
            bm, bh = base_for(base, k)
            for s in sigmas:
                c = cell(select(noise, k=k, sigma=s, attacker=a))
                if c is None:
                    continue
                h = c["half"] if np.isfinite(c["half"]) else np.inf
                comb = float(np.hypot(h, bh))
                diff = c["mean"] - bm
                out[a][(k, s)] = {"diff": diff, "comb": comb, "clear": diff - comb > 0, "cell": c}
    return out


def stopping(noise, base, ks, sigmas):
    """Lowest tested sigma from which every larger tested sigma is stopped."""
    rows = []
    for a, _ in ATTACKERS:
        for k in ks:
            bm, _ = base_for(base, k)
            cells = [(s, cell(select(noise, k=k, sigma=s, attacker=a))) for s in sigmas]
            cells = [(s, c) for s, c in cells if c is not None]
            if not cells:
                continue

            def first_stop(ok):
                stop = None
                for s, c in reversed(cells):
                    if ok(c):
                        stop = s
                    else:
                        break
                return stop
            by_ssim = first_stop(lambda c: np.isfinite(c["half"]) and c["mean"] + c["half"] <= bm)
            by_succ = first_stop(lambda c: c["success"] <= 0.01)
            rows.append((a, k, bm, by_ssim, by_succ))
    return rows


def decomposition(noise, matched, ks, sigmas):
    rows = []
    for a, _ in ATTACKERS:
        for k in [k for k in ks if k > 1]:
            mix_only = cell(select(noise, k=k, sigma=0.0, attacker=a))
            for s in sigmas:
                ksum = cell(select(noise, k=k, sigma=s, attacker=a))
                informed = cell(select(noise, k=1, sigma=s, attacker=a))
                if s == 0:
                    msnr = informed  # zero noise: the matched single example is the
                else:
                    msnr = cell(select(matched, k=k, sigma=s, attacker=a))
                get = lambda c: c["mean"] if c else np.nan  # noqa: E731
                rows.append({"attacker": a, "k": k, "sigma": s, "ksum": ksum, "informed": informed,
                             "matched": msnr, "mix_only": mix_only,
                             "benefit": get(msnr) - get(informed),
                             "cost": get(msnr) - get(ksum),
                             "net": get(ksum) - get(informed)})
    return rows


def fig_heatmap(grid, ks, sigmas, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.65)
    for ax, (a, label) in zip(axes, ATTACKERS):
        m = np.full((len(ks), len(sigmas)), np.nan)
        for i, k in enumerate(ks):
            for j, s in enumerate(sigmas):
                if (k, s) in grid[a]:
                    m[i, j] = grid[a][(k, s)]["diff"]
        im = ax.imshow(m, cmap="RdBu_r", norm=norm, aspect="auto")
        for i, k in enumerate(ks):
            for j, s in enumerate(sigmas):
                g = grid[a].get((k, s))
                if g is None:
                    continue
                txt = f"{g['diff']:+.2f}" + ("*" if g["clear"] else "")
                strong = abs(g["diff"]) > 0.35
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                        color="white" if strong else INK, fontweight="bold" if g["clear"] else "normal")
        ax.set_xticks(range(len(sigmas)), [sig_label(s) for s in sigmas])
        ax.set_yticks(range(len(ks)), [str(k) for k in ks])
        ax.set_xlabel("Noise multiplier σ on the sum")
        ax.set_title(label)
        ax.grid(False)
    axes[0].set_ylabel("K, examples in the sum")
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cb.set_label("Mean SSIM minus no-information baseline")
    fig.suptitle("Leak above the no-information baseline (* = 95% CI above 0)", x=0.45, color=INK)
    fig.savefig(path)
    plt.close(fig)


def fig_decomposition(noise, matched, base, sigmas, path, ks=(4, 8)):
    fig, axes = plt.subplots(2, len(ks), figsize=(10, 6.4), sharex=True, sharey=True)
    x = np.arange(len(sigmas))
    for r, (a, label) in enumerate(ATTACKERS):
        for c, k in enumerate(ks):
            ax = axes[r, c]
            bm, bh = base_for(base, k)
            ax.axhspan(bm - bh, bm + bh, color=MUTED, alpha=0.12, lw=0)
            ax.axhline(bm, color=MUTED, ls="--", lw=1.2, label=f"No information (best-matched, K={k})")
            series = [("Sum of K examples", select(noise, k=k, attacker=a), SERIES[0], MARKERS[0]),
                      ("Informed: knows the other K−1 (= K=1)", select(noise, k=1, attacker=a),
                       SERIES[1], MARKERS[1]),
                      ("Matched SNR: one example, same relative noise", select(matched, k=k, attacker=a),
                       SERIES[2], MARKERS[2])]
            for name, rows, color, marker in series:
                pts = []
                for j, s in enumerate(sigmas):
                    if name.startswith("Matched") and s == 0:
                        cc = cell(select(noise, k=1, sigma=0.0, attacker=a))
                    else:
                        cc = cell([q for q in rows if q["sigma"] == s])
                    if cc:
                        pts.append((j, cc["mean"], cc["half"] if np.isfinite(cc["half"]) else 0))
                if len(pts) <= 1 and name.startswith("Matched"):
                    continue
                if pts:
                    px, py, ph = map(np.array, zip(*pts))
                    ax.plot(px, py, color=color, marker=marker, label=name)
                    ax.fill_between(px, np.clip(py - ph, 0, 1), np.clip(py + ph, 0, 1),
                                    color=color, alpha=0.15, lw=0)
            ax.set_title(f"{label}, K = {k}")
            ax.set_ylim(-0.02, 1.02)
            if r == 1:
                ax.set_xticks(x, [sig_label(s) for s in sigmas])
                ax.set_xlabel("Noise multiplier σ on the sum")
            if c == 0:
                ax.set_ylabel("Mean SSIM")
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels and not l.startswith("No information"):
                handles.append(h)
                labels.append(l)
    handles.append(plt.Line2D([], [], color=MUTED, ls="--"))
    labels.append("No-information baseline (best-matched unrelated digits)")
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Where does the protection come from? Sum vs informed vs matched-SNR", color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    style()
    noise = load_records("noise_s*.json")
    matched = load_records("matched_s*.json")
    if not noise:
        sys.exit("No results/runs/noise_s*.json yet; run scripts/attack_noise_sum.py first.")
    base = load_baselines()
    ks = sorted({r["k"] for r in noise})
    sigmas = sorted({r["sigma"] for r in noise})
    seeds = sorted({r["seed"] for r in noise})
    mseeds = sorted({r["seed"] for r in matched})
    prelim = len(seeds) < 5 or not matched

    grid = leak_grid(noise, base, ks, sigmas)
    stops = stopping(noise, base, ks, sigmas)
    decomp = decomposition(noise, matched, ks, sigmas)
    kap = kappa_estimate()
    kappa = kap["kappa_hat"] if kap["kappa_hat"] > 0 else KAPPA_FLOOR

    L = ["# RQ4: Signal Against Noise, or Mixing?", ""]
    if prelim:
        L += [f"> **PRELIMINARY.** Noise grid seeds: {seeds or 'none'}. Matched-SNR seeds: "
              f"{mseeds or 'none yet'}. Rerun `python scripts/analyze_rq4.py` when all seeds land.", ""]
    if len(seeds) < 2:
        L += ["> With one seed, every ± below is a 95% interval over the sums in that seed. It "
              "ignores seed-to-seed variation, so it is too narrow. Do not quote it.", ""]
    L += ["MNIST, fresh LeNet, clip C = 30, Gaussian noise σ·C added once to the sum of K clipped "
          "per-example gradients, attacker given the true labels, images scored against their "
          "best-matched original. Baseline = unrelated real digits, best-matched, at the same K "
          "(`results/baselines.md`).", ""]

    # 1.
    L += ["## 1. Leak Above the No-Information Baseline", "",
          "Mean SSIM minus the baseline at the same K. **Bold** = clear leak: the difference minus "
          "its 95% interval (attack and baseline intervals combined) is still above 0. "
          "Figure: `results/fig_rq4_heatmap.png`.", ""]
    for a, label in ATTACKERS:
        L += [f"**{label}**", "", "| K | " + " | ".join(f"σ={sig_label(s)}" for s in sigmas) + " |",
              "|---|" + "---|" * len(sigmas)]
        for k in ks:
            cells = []
            for s in sigmas:
                g = grid[a].get((k, s))
                if g is None:
                    cells.append("—")
                else:
                    v = f"{g['diff']:+.3f}"
                    cells.append(f"**{v}**" if g["clear"] else v)
            L.append(f"| {k} | " + " | ".join(cells) + " |")
        L.append("")

    # 2.
    L += ["## 2. Stopping Noise σ*(K)", "",
          "The lowest tested σ from which the attack is stopped at that σ and at every larger "
          "tested σ. Two rules:",
          "- **SSIM rule:** the upper end of the attack's 95% interval is at or "
          "below the baseline.",
          "- **Success rule:** at most 1% of images reach SSIM ≥ 0.9.", "",
          "| Attacker | K | Baseline SSIM | σ* (SSIM rule) | ε at σ* | σ* (success rule) | ε at σ* |",
          "|---|---|---|---|---|---|---|"]
    from scripts.attack_noise_sum import epsilon_for
    for a, k, bm, s1, s2 in stops:
        e = lambda s: "—" if s is None else ("∞" if s == 0 else f"{epsilon_for(s):.3g}")  # noqa: E731
        f = lambda s: ("not stopped" if s is None  # noqa: E731
                       else "0 (stopped with no noise)" if s == 0 else sig_label(s))
        L.append(f"| {a} | {k} | {bm:.3f} | {f(s1)} | {e(s1)} | {f(s2)} | {e(s2)} |")
    L += ["", "ε is for one Gaussian release (δ = 1e-5, sampling rate 1, one step). It is not "
          "the DP-SGD training ε.", ""]

    # 3.
    L += ["## 3. Pulling the Two Effects Apart", "",
          "Four attacks on the same images at the same σ:",
          "- **Sum:** the attack on the noisy sum of K examples.",
          "- **Informed:** an attacker who already knows the other K−1 examples. It subtracts "
          "their gradients and is left with one example plus the same noise. For a single step on "
          "a fresh model this is exactly the K = 1 attack at the same σ, so we reuse the K = 1 "
          "results.",
          "- **Matched SNR:** one example attacked alone, with its noise shrunk so that its "
          "noise-to-signal ratio equals what that example has inside the K-sum "
          "(σ′ = σ·‖gᵢ‖ / ‖Σⱼ gⱼ‖). From `scripts/attack_matched_snr.py`.",
          "- **Mixing only:** the K-sum with no noise.", "",
          "Two numbers, in plain words:",
          "- **Signal benefit = Matched SNR − Informed.** How much the bigger summed signal "
          "weakens the noise, for one example. Positive means the bigger sum helps the attacker.",
          "- **Mixing cost = Matched SNR − Sum.** How much harder it is to pull one example out "
          "of a mix of K, at the same noise-to-signal ratio. Positive means mixing protects.",
          "- **Net = Sum − Informed = Signal benefit − Mixing cost.** Positive means the signal "
          "effect wins; negative means mixing wins.", ""]
    if not matched:
        L += ["> Matched-SNR results are not in yet, so the signal benefit and mixing cost are "
              "blank. Only the net effect can be read now.", ""]
    L += ["| Attacker | K | σ | Sum | Informed (K=1) | Matched SNR | Mixing only (σ=0) "
          "| Signal benefit | Mixing cost | Net |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    num = lambda v: "—" if not np.isfinite(v) else f"{v:+.3f}"  # noqa: E731
    for d in decomp:
        L.append(f"| {d['attacker']} | {d['k']} | {sig_label(d['sigma'])} | {fmt(d['ksum'])} | "
                 f"{fmt(d['informed'])} | {fmt(d['matched'])} | {fmt(d['mix_only'])} | "
                 f"{num(d['benefit'])} | {num(d['cost'])} | {num(d['net'])} |")
    L += ["", "Figure: `results/fig_rq4_decomposition.png` (K = 4 and 8).", ""]

    # 4.
    L += ["## 4. Measured Success vs the Formal Bound", "",
          "Reconstruction robustness (ReRo) bound for one release of the Gaussian mechanism "
          "(Hayes et al. 2023, arXiv:2302.07225, Corollary 4, with q = 1 and T = 1):", "",
          "  success ≤ Φ(Φ⁻¹(κ) + 1/σ) = 1 − Φ(Φ⁻¹(1 − κ) − 1/σ)", "",
          "- The paper gives the general bound as a blow-up function. The closed form above, "
          "for one full-batch step, is **our own reduction** (a Gaussian mean shift of 1/σ). "
          "Check it before citing.",
          f"- κ is the best success an attacker can get **without** seeing any gradient: the best "
          f"single fixed guess, scored on random MNIST test digits at SSIM ≥ {SUCCESS}. We measured "
          f"κ̂ = {kap['kappa_hat']:.4g} over {kap['targets']} targets and {kap['guesses']} guesses "
          f"(random training digits, the average digit, all black). Taking the best of many "
          f"guesses makes κ̂ slightly too high, which only loosens the bound."
          + (f" That is 0, so we use a floor of κ = {KAPPA_FLOOR:g}. The true κ is likely smaller, "
             "which would only make the bound tighter." if kap["kappa_hat"] <= 0 else ""),
          "- The bound covers the informed attacker (who knows the other K−1 examples), so it "
          "covers every K. Measured success is shown for K = 1 (the informed attacker) and for "
          "the highest across all K.",
          "- The bound is about any attacker. Our attacks are only two attackers, so measured "
          "success below the bound is expected; the gap shows how loose the bound is in practice, "
          "or how much better an attacker could do.", "",
          "| σ | ε (one release) | ReRo bound | Basic, K=1 | Cosine, K=1 | Highest over K and attackers |",
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
        bound = rero_bound(s, kappa)
        L.append(f"| {sig_label(s)} | {eps} | {bound * 100:.1f}% | {meas['basic']} | "
                 f"{meas['cosine']} | {hi} |")
    all_sig = sorted(set(sigmas) | set(BOUND_EXTRA_SIGMAS))
    useful = [s for s in all_sig if s > 0 and rero_bound(s, kappa) < 0.99]
    failed = [s for s in sigmas if s > 0 and all(
        c["success"] <= 0.01 for c in (cell(select(noise, k=k, sigma=s, attacker=a))
                                       for k in ks for a, _ in ATTACKERS) if c)]
    L += [""]
    if useful:
        L.append(f"The bound first drops below 99% at σ = {sig_label(useful[0])} "
                 f"(ε = {epsilon_for(useful[0]):.3g}).")
    if failed:
        L.append(f"Both attacks already fail (≤ 1% success at every K) from σ = {sig_label(failed[0])} "
                 f"(ε = {epsilon_for(failed[0]):.3g}), where the bound guarantees nothing.")
    L += [""]

    with open(os.path.join(RES, "rq4_summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    fig_heatmap(grid, ks, sigmas, os.path.join(RES, "fig_rq4_heatmap.png"))
    fig_decomposition(noise, matched, base, sigmas, os.path.join(RES, "fig_rq4_decomposition.png"))
    print("\n".join(L))
    print("Saved results/rq4_summary.md, results/fig_rq4_heatmap.png, results/fig_rq4_decomposition.png")


if __name__ == "__main__":
    main()
