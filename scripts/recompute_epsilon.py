"""Recompute the one-release epsilon under replace-one adjacency."""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.analyze_rq4 import rero_bound
from scripts.attack_noise_sum import DEFAULT_SIGMAS, epsilon_for

DELTA = 1e-5
# Sigma values beyond the study grid that the paper's bound uses: sigma = 1
PAPER_SIGMAS = [0.3, 1.0, 2.0]


def epsilon_replace_one(sigma, delta=DELTA):
    """One-release epsilon when the clipped sum has sensitivity 2C (noise std sigma*C)."""
    return epsilon_for(sigma / 2, delta) if sigma > 0 else float("inf")


def rero_replace_one(sigma, kappa):
    """The ReRo bound when the clipped sum has sensitivity 2C: the add/remove bound at sigma/2."""
    return rero_bound(sigma / 2, kappa) if sigma > 0 else 1.0


def study_sigmas(root):
    """Every sigma and delta recorded in the saved noise runs."""
    sigmas, deltas, saved = set(), set(), {}
    for path in sorted(glob.glob(os.path.join(root, "*", "noise_s[0-9]*.json"))):
        with open(path) as f:
            data = json.load(f)
        sigmas.update(float(s) for s in data["args"]["sigmas"])
        for r in data["records"]:
            if r["sigma"] > 0 and r.get("epsilon") is not None:
                saved.setdefault(float(r["sigma"]), set()).add(round(float(r["epsilon"]), 6))
    return sigmas, saved


def fmt(e):
    if not np.isfinite(e):
        return "∞"
    return f"{e:.3g}" if e < 1000 else f"{e:.2e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/study")
    p.add_argument("--out", default="results/study/epsilon_replace_one.md")
    args = p.parse_args()

    from_study, saved = study_sigmas(args.root)
    sigmas = sorted(from_study | set(DEFAULT_SIGMAS) | set(PAPER_SIGMAS))
    # The saved epsilons must equal the old formula, so we know what "old" was.
    for s, eps in saved.items():
        assert all(abs(e - epsilon_for(s, DELTA)) <= 1e-6 * max(1.0, e) for e in eps), (s, eps)

    def where(s):
        if s in from_study:
            return "study grid (all settings) and paper"
        if s in PAPER_SIGMAS:
            return "reconstruction bound (analyze_rq4.py; σ = 1 is in the paper)"
        return "script default only (not in the study)"

    lines = ["# One-Release ε Under Replace-One Adjacency", "",
             "Made by `scripts/recompute_epsilon.py` from the saved results; no attack was rerun and no "
             "result file was changed.", "",
             "- **Mechanism:** clip each example's gradient to norm C, sum the K clipped gradients, add "
             "Gaussian noise N(0, (σC)²) once, divide by K (`scripts/attack_noise_sum.py`).",
             "- **Old ε (add/remove one example):** sensitivity C, so the accountant gets noise multiplier σ. "
             "This is what the study JSON, `summary.md` and the earlier paper draft report.",
             "- **Replace-one ε:** one example is swapped for another, so K is unchanged and can be public. "
             "The sum moves by up to 2C, so the noise is σ/2 times the sensitivity: the accountant gets σ/2.",
             f"- Same accountant as before: Opacus RDP, sampling rate 1, one step, δ = {DELTA:g}. "
             "ε is an accounting upper bound for one release; it is not the DP-SGD training ε.",
             "- Replace-one ε at σ equals add/remove ε at σ/2. All saved ε values were checked to match the "
             "old formula.", "",
             "| σ | Old ε (add/remove, σ) | Replace-one ε (σ/2) | Ratio | Used in |",
             "|---|---|---|---|---|"]
    for s in sigmas:
        old, new = epsilon_for(s, DELTA), epsilon_replace_one(s, DELTA)
        ratio = f"{new / old:.2f}×" if np.isfinite(old) and old > 0 else "—"
        lines.append(f"| {s:g} | {fmt(old)} | {fmt(new)} | {ratio} | {where(s)} |")
    lines += ["", "## What changes in the reported numbers", "",
              "- σ = 0.01 stops every single-step attack. Its one-release ε goes from "
              f"{fmt(epsilon_for(0.01))} to {fmt(epsilon_replace_one(0.01))}.",
              f"- The bound's σ = 1 goes from ε = {fmt(epsilon_for(1.0))} to {fmt(epsilon_replace_one(1.0))}.",
              "- For tiny σ, RDP ε grows like 1/σ², so halving σ multiplies ε by about 4. Near σ = 1 the "
              "ratio is smaller.",
              f"- Check: replace-one ε at σ = 2 ({fmt(epsilon_replace_one(2.0))}) equals the old ε at σ = 1.",
              "- The earlier paper draft already said the replace-one values would need twice the σ; this "
              "table gives those values."]
    kappa_path = os.path.join(args.root, "kappa_mnist.json")
    if os.path.exists(kappa_path):
        with open(kappa_path) as f:
            kappa = float(json.load(f)["kappa_hat"])
        lines += ["", "## The illustrative reconstruction bound (MNIST)", "",
                  "ReRo bound for one release (`scripts/analyze_rq4.py`, Kaissis et al. 2023 closed form): "
                  "success ≤ Φ(Φ⁻¹(κ) + Δ/σ), with κ̂ = "
                  f"{kappa:.4f} (`{kappa_path}`). Add/remove uses Δ = 1 (what the earlier draft reports). "
                  "Replace-one uses Δ = 2, which is the add/remove bound at σ/2.", "",
                  "| σ | Bound, add/remove (Δ = 1) | Bound, replace-one (Δ = 2) |", "|---|---|---|"]
        for s in PAPER_SIGMAS:
            lines.append(f"| {s:g} | {100 * rero_bound(s, kappa):.1f}% | {100 * rero_replace_one(s, kappa):.1f}% |")
        lines += ["", "- Which value to report: the cited ReRo results (Hayes et al. 2023; Kaissis et al. 2023) "
                  "compare the release that holds the target with a reference release that does not depend on "
                  "the target (add-one). Any such reference works, so for a fixed-size release (public K) one "
                  "can take the other K − 1 clipped gradients plus the same noise, with the same divisor K. The "
                  "target then shifts the mean by at most C/K against noise σC/K, so Δ = 1 holds and the "
                  "add/remove column is a valid bound. The replace-one column (two candidate targets) is a "
                  "valid but looser bound. Report the add/remove value (4.4% at σ = 1) with this argument.",
                  "- This is separate from the ε adjacency above: the one-release ε of a fixed-size release is "
                  "still stated under replace-one.",
                  "- κ̂ is a lower estimate from finitely many guesses, so both bounds are indicative, "
                  "not certified."]
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
