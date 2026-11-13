"""Compare the four study settings: where the sum leaks, and how much noise stops it."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.analyze_rq4 import (ATTACKERS, SETTING_TITLES, analyze, clear,  # noqa: E402
                                 fmt, load_context, sig_label, stopped)
from src.plotting import INK, MARKERS, MUTED, SERIES, plt, style  # noqa: E402

ORDER = ["mnist-fresh", "mnist-trained", "cifar10-fresh", "mnist-fedavg"]
SHORT = {"mnist-fresh": "MNIST, fresh", "mnist-trained": "MNIST, trained",
         "cifar10-fresh": "CIFAR-10, fresh", "mnist-fedavg": "MNIST, FedAvg update"}
MID_SIGMA = 1e-3  # the noise level shown in the informed-attacker


def verdict(d):
    if d is None:
        return "—"
    if clear(d):
        return "leak"
    if stopped(d):
        return "below"
    return "unclear"


def leak_cell(d_ssim, d_psnr):
    """'**leak** +0.45 / +12.3 dB': verdict from SSIM, then SSIM and PSNR differences."""
    if d_ssim is None:
        return "—"
    v = verdict(d_ssim)
    word = {"leak": "**leak**", "below": "*at or below line*", "unclear": "no clear leak"}[v]
    psnr = "" if d_psnr is None else f" / {d_psnr['mean']:+.1f} dB" + (" ✓" if verdict(d_psnr) == v else " ✗")
    return f"{word} {d_ssim['mean']:+.2f}{psnr}"


def stop_text(s):
    from scripts.attack_noise_sum import epsilon_for
    if s is None:
        return "not stopped"
    if s == 0:
        return "0 (no noise needed)"
    return f"{sig_label(s)} (ε {epsilon_for(s):.2g})"


def load_all(root, models_dir, names):
    loaded, missing = {}, []
    for name in names:
        ctx = load_context(name, root, models_dir)
        if not ctx["noise"] or ctx["base"] is None:
            missing.append(name)
            continue
        loaded[name] = (ctx, analyze(ctx))
    return loaded, missing


def summary_lines(loaded, missing):
    L = ["# Study Summary: Four Settings", "",
         "Made by `scripts/compare_settings.py` from `results/study/`. Each setting has its own "
         "full report in `results/study/<setting>/rq4_summary.md`.", "",
         "**How to read this**",
         "- The attacker sees the **sum** of K images' updates, with or without noise.",
         "- A score only counts as a **leak** if it is clearly above what a no-information "
         "guess scores. The guess knows the labels, like the attacker does "
         "(the **label-aware baseline**).",
         "- \"Clearly\" means the whole 95% interval over seeds is above that line.",
         "- SSIM is the standard form. PSNR is a second check.", ""]
    if missing:
        L += [f"> Not in yet: {', '.join(missing)}.", ""]

    # Settings table
    L += ["## The Settings", "",
          "| Setting | What is attacked | Clip C | Cosine TV | Model accuracy | Seeds |",
          "|---|---|---|---|---|---|"]
    for name, (ctx, R) in loaded.items():
        a = ctx["run_args"]
        clip = f"{a['clip']:.3g}" if ctx["update"] == "gradient" and a.get("clip") else "— (no noise)"
        tv = f"{a['cosine_tv']:g}" if a.get("cosine_tv") is not None else "—"
        acc = (", ".join(f"{v:.3f}" for _, v in sorted(ctx["accuracy"].items()))
               if ctx["accuracy"] else ("fresh (untrained)" if not ctx["trained"] else "not found"))
        L.append(f"| {name} | {ctx['title']} | {clip} | {tv} | {acc} | {R['seeds']} |")
    L += ["", "Clip C and the cosine TV weight were chosen on held-out images "
          "(`scripts/tune_attack.py`). C sits just above the largest per-example gradient norm, so "
          "clipping almost never acts and the noise sweep measures noise only.", ""]

    # 1.
    ks_all = sorted({k for _, R in loaded.values() for k in R["ks_leak"]})
    L += ["## 1. Does the Sum Leak? (No Noise)", "",
          "Each cell: verdict from SSIM, then SSIM minus baseline / PSNR minus baseline. "
          "✓ = PSNR gives the same verdict, ✗ = it does not. \"At or below line\" is not proof "
          "that nothing leaked: averages hide single images that do leak.", "",
          "| Setting | Attacker | " + " | ".join(f"K={k}" for k in ks_all) + " |",
          "|---|---|" + "---|" * len(ks_all)]
    for name, (ctx, R) in loaded.items():
        for att, _ in ATTACKERS:
            cells = [leak_cell(R["grid"][att].get((k, 0.0)), R["grid_psnr"][att].get((k, 0.0))) for k in ks_all]
            L.append(f"| {SHORT.get(name, name)} | {att} | " + " | ".join(cells) + " |")
    L.append("")

    # 2.
    noisy = {n: v for n, v in loaded.items() if v[1]["has_noise"]}
    if noisy:
        ks = sorted({k for _, R in noisy.values() for k in R["ks"]})
        L += ["## 2. How Much Noise Stops the Attack?", "",
              "The lowest tested noise σ from which the attack is stopped (the whole interval at or "
              "below the baseline) at that σ and every larger one. ε is for **one** release of the "
              "noisy sum (δ = 1e-5); it is not the training ε. \"Not stopped\" means even the "
              "largest tested σ did not stop it.", "",
              "| Setting | Attacker | " + " | ".join(f"K={k}" for k in ks) + " |",
              "|---|---|" + "---|" * len(ks)]
        for name, (ctx, R) in noisy.items():
            by = {(a, k): s1 for a, k, s1, _, _ in R["stops"]}
            for att, _ in ATTACKERS:
                L.append(f"| {SHORT.get(name, name)} | {att} | "
                         + " | ".join(stop_text(by.get((att, k))) if (att, k) in by else "—" for k in ks) + " |")
        L += ["", "PSNR cross-check of the same rule:", "",
              "| Setting | Attacker | " + " | ".join(f"K={k}" for k in ks) + " |",
              "|---|---|" + "---|" * len(ks)]
        for name, (ctx, R) in noisy.items():
            by = {(a, k): s2 for a, k, _, s2, _ in R["stops"]}
            for att, _ in ATTACKERS:
                L.append(f"| {SHORT.get(name, name)} | {att} | "
                         + " | ".join(stop_text(by.get((att, k))) if (att, k) in by else "—" for k in ks) + " |")
        L.append("")

        # 3.
        L += ["## 3. The Sum vs an Attacker Who Knows the Other Images", "",
              "Net = SSIM of the K-sum minus SSIM of an attacker who knows the other K−1 images "
              "(the K = 1 attack at the same σ). An optimal attacker can never do worse when it "
              "knows more, so net ≤ 0 is expected. What matters is **how big the gap is**. "
              "**Bold** = the 95% interval excludes 0.", "",
              f"| Setting | Attacker | K | Net, no noise | Net, σ = {sig_label(MID_SIGMA)} |",
              "|---|---|---|---|---|"]
        for name, (ctx, R) in noisy.items():
            nets = {(d["attacker"], d["k"], d["sigma"]): d["net"] for d in R["decomp"]}
            for att, _ in ATTACKERS:
                for k in [k for k in R["ks"] if k > 1]:
                    vals = []
                    for s in (0.0, MID_SIGMA):
                        d = nets.get((att, k, s))
                        v = fmt(d, signed=True)
                        vals.append(f"**{v}**" if (clear(d) or stopped(d)) else v)
                    L.append(f"| {SHORT.get(name, name)} | {att} | {k} | {vals[0]} | {vals[1]} |")
        L.append("")

    # 4.
    longs = [(name, R) for name, (ctx, R) in loaded.items() if R["long"]]
    if longs:
        L += ["## 4. Did the Attack Just Run Out of Steps?", "",
              "The same attack with 4× the steps, no noise.", "",
              "| Setting | Attacker | K | Normal − baseline | 4× steps − baseline | 4× minus normal |",
              "|---|---|---|---|---|---|"]
        for name, R in longs:
            for d in R["long"]:
                L.append(f"| {SHORT.get(name, name)} | {d['attacker']} | {d['k']} | "
                         f"{fmt(d['normal_leak'], signed=True)} | {fmt(d['long_leak'], signed=True)} | "
                         f"{fmt(d['gain'], signed=True)} |")
        L.append("")

    # 5.
    L += ["## 5. In Plain Words", ""]
    for name, (ctx, R) in loaded.items():
        for att, label in ATTACKERS:
            leak_ks = [k for k in R["ks_leak"] if clear(R["grid"][att].get((k, 0.0)))]
            no_ks = [k for k in R["ks_leak"] if not clear(R["grid"][att].get((k, 0.0)))]
            part = (f"leaks clearly at K = {', '.join(map(str, leak_ks))}" if leak_ks
                    else "shows no clear leak at any K")
            if leak_ks and no_ks:
                part += f"; no clear leak on average at K = {', '.join(map(str, no_ks))}"
            line = f"- **{SHORT.get(name, name)}, {att}:** {part}"
            if R["has_noise"]:
                stops = {k: s1 for a, k, s1, _, _ in R["stops"] if a == att}
                known = [s for s in stops.values() if s is not None]
                if stops and all(s is not None for s in stops.values()):
                    line += f". Noise σ ≥ {sig_label(max(known))} stops it at every tested K"
                elif known:
                    line += (f". Noise stops it at some K (σ up to {sig_label(max(known))}), but not at "
                             f"K = {', '.join(str(k) for k, s in stops.items() if s is None)}")
                else:
                    line += ". No tested noise level stops it"
            L.append(line + ".")
    L += ["", "- A low **average** score does not mean no image leaked. Look at the example images "
          "too: single images can be clearly recovered in a sum the average calls safe.",
          "- The ε values that stop these attacks are very large. Stopping two practical attacks is "
          "not a privacy guarantee.", "",
          "## Figures", "",
          "- `results/fig_study_leak_vs_k.png`: leak vs K for every setting, no noise.",
          "- `results/fig_study_stopping_noise.png`: stopping noise vs K, with ε.",
          "- `results/study/<setting>/fig_rq4_heatmap.png`: leak over K and σ, per setting.", ""]
    return L


def fig_leak_vs_k(loaded, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    names = list(loaded)
    for ax, (att, label) in zip(axes, ATTACKERS):
        for i, name in enumerate(names):
            ctx, R = loaded[name]
            pts = [(k, R["grid"][att][(k, 0.0)]) for k in R["ks_leak"] if R["grid"][att].get((k, 0.0))]
            if not pts:
                continue
            jitter = 2 ** ((i - (len(names) - 1) / 2) * 0.04)
            x = np.array([k * jitter for k, _ in pts])
            m = np.array([d["mean"] for _, d in pts])
            h = np.array([d["half"] if np.isfinite(d["half"]) else 0 for _, d in pts])
            ax.errorbar(x, m, yerr=h, color=SERIES[i % len(SERIES)], marker=MARKERS[i % len(MARKERS)],
                        capsize=3, label=SHORT.get(name, name))
            for (k, d), xi, mi in zip(pts, x, m):
                if clear(d):
                    ax.plot(xi, mi, marker=MARKERS[i % len(MARKERS)], color=SERIES[i % len(SERIES)],
                            ms=9, mfc="white", zorder=0)
        ax.axhline(0, color=INK, lw=1.2, ls="--")
        ax.text(0.01, 0.01, "no-information line (knows labels)", transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=7, color=MUTED)
        ks = sorted({k for _, R in loaded.values() for k in R["ks_leak"]})
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("K, images summed into one update")
        ax.set_title(label)
    axes[0].set_ylabel("Mean SSIM minus label-aware baseline\n(standard SSIM, 95% CI over seeds)")
    axes[1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("Does the sum leak without noise? Above 0 = more than a no-information guess\n"
                 "(ring = clear leak: the whole 95% interval is above 0)", color=INK, y=1.03)
    fig.savefig(path)
    plt.close(fig)


def fig_stopping(loaded, path):
    from scripts.attack_noise_sum import epsilon_for
    noisy = {n: v for n, v in loaded.items() if v[1]["has_noise"]}
    if not noisy:
        return False
    sig_all = sorted({s for _, R in noisy.values() for s in R["sigmas"] if s > 0})
    top, bottom = sig_all[-1] * 3, sig_all[0] / 3
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, (att, label) in zip(axes, ATTACKERS):
        for i, name in enumerate(noisy):
            ctx, R = noisy[name]
            rows = sorted((k, s1) for a, k, s1, _, _ in R["stops"] if a == att)
            if not rows:
                continue
            c, mk = SERIES[i % len(SERIES)], MARKERS[i % len(MARKERS)]
            jitter = 2 ** ((i - (len(noisy) - 1) / 2) * 0.05)
            xs = np.array([k * jitter for k, _ in rows])
            ys = np.array([top if s is None else (bottom if s == 0 else s) for _, s in rows])
            ax.plot(xs, ys, color=c, marker=mk, label=SHORT.get(name, name))
            for (k, s), x, y in zip(rows, xs, ys):
                if s is None:
                    ax.plot(x, y, marker="^", color=c, ms=10, mfc="white")
                elif s == 0:
                    ax.plot(x, y, marker="v", color=c, ms=10, mfc="white")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ks = sorted({k for _, R in noisy.values() for k in R["ks"]})
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_ylim(bottom / 1.5, top * 1.5)
        ax.axhline(top, color=MUTED, lw=0.8, ls=":")
        ax.axhline(bottom, color=MUTED, lw=0.8, ls=":")
        ax.text(0.01, top, " △ not stopped by any tested σ", transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=7, color=MUTED)
        ax.text(0.01, bottom, " ▽ stopped with no noise", transform=ax.get_yaxis_transform(),
                ha="left", va="top", fontsize=7, color=MUTED)
        ax.set_xlabel("K, images summed into one update")
        ax.set_title(label)
        twin = ax.twinx()
        twin.set_yscale("log")
        twin.set_ylim(ax.get_ylim())
        twin.set_yticks(sig_all)
        twin.set_yticklabels([f"{epsilon_for(s):.2g}" for s in sig_all], fontsize=7)
        twin.minorticks_off()
        twin.grid(False)
        if ax is axes[-1]:
            twin.set_ylabel("ε for one release (δ = 1e-5)")
    axes[0].set_yticks(sig_all)
    axes[0].set_yticklabels([sig_label(s) for s in sig_all])
    axes[0].minorticks_off()
    axes[0].set_ylabel("Stopping noise σ* (SSIM rule)")
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.12, 1.0))
    fig.suptitle("How much noise on the sum stops the attack?", color=INK)
    fig.savefig(path)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/study")
    p.add_argument("--models-dir", default="results/models")
    p.add_argument("--settings", nargs="+", default=ORDER)
    args = p.parse_args()
    style()
    loaded, missing = load_all(args.root, args.models_dir, args.settings)
    if not loaded:
        sys.exit(f"No settings with results under {args.root} yet.")
    fig_dir = "results" if os.path.normpath(args.root) == os.path.normpath("results/study") else args.root
    L = summary_lines(loaded, missing)
    with open(os.path.join(args.root, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    leak_fig = os.path.join(fig_dir, "fig_study_leak_vs_k.png")
    stop_fig = os.path.join(fig_dir, "fig_study_stopping_noise.png")
    fig_leak_vs_k(loaded, leak_fig)
    made = [leak_fig] + ([stop_fig] if fig_stopping(loaded, stop_fig) else [])
    print("\n".join(L))
    print(f"Saved {os.path.join(args.root, 'summary.md')}, " + ", ".join(made))


if __name__ == "__main__":
    main()
