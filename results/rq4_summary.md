# RQ4: Signal Against Noise, or Mixing?

MNIST, fresh LeNet, clip C = 30, Gaussian noise σ·C added once to the sum of K clipped per-example gradients, attacker given the true labels, images scored against their best-matched original. Baseline = unrelated real digits, best-matched, at the same K (`results/baselines.md`).

## 1. Leak Above the No-Information Baseline

Mean SSIM minus the baseline at the same K. **Bold** = clear leak: the difference minus its 95% interval (attack and baseline intervals combined) is still above 0. Figure: `results/fig_rq4_heatmap.png`.

**Basic (iDLG, L-BFGS)**

| K | σ=0 | σ=3e-5 | σ=1e-4 | σ=3e-4 | σ=0.001 | σ=0.003 | σ=0.01 | σ=0.1 |
|---|---|---|---|---|---|---|---|---|
| 1 | **+0.597** | **+0.490** | **+0.270** | **+0.071** | -0.099 | -0.314 | -0.338 | -0.363 |
| 2 | **+0.535** | **+0.369** | **+0.133** | -0.041 | -0.240 | -0.350 | -0.369 | -0.385 |
| 4 | **+0.398** | **+0.192** | -0.020 | -0.181 | -0.347 | -0.377 | -0.391 | -0.400 |
| 8 | +0.034 | -0.058 | -0.208 | -0.363 | -0.428 | -0.436 | -0.442 | -0.451 |

**Cosine (Geiping et al.)**

| K | σ=0 | σ=3e-5 | σ=1e-4 | σ=3e-4 | σ=0.001 | σ=0.003 | σ=0.01 | σ=0.1 |
|---|---|---|---|---|---|---|---|---|
| 1 | **+0.590** | **+0.567** | **+0.533** | **+0.407** | **+0.042** | -0.154 | -0.296 | -0.361 |
| 2 | **+0.339** | **+0.338** | **+0.352** | **+0.292** | +0.038 | -0.198 | -0.336 | -0.383 |
| 4 | +0.192 | +0.185 | +0.195 | +0.184 | +0.029 | -0.217 | -0.358 | -0.399 |
| 8 | -0.005 | -0.020 | -0.013 | -0.018 | -0.076 | -0.249 | -0.411 | -0.449 |

## 2. Stopping Noise σ*(K)

The lowest tested σ from which the attack is stopped at that σ and at every larger tested σ. Two rules:
- **SSIM rule:** the upper end of the attack's 95% interval is at or below the baseline.
- **Success rule:** at most 1% of images reach SSIM ≥ 0.9.

| Attacker | K | Baseline SSIM | σ* (SSIM rule) | ε at σ* | σ* (success rule) | ε at σ* |
|---|---|---|---|---|---|---|
| basic | 1 | 0.368 | 0.001 | 5.5e+05 | 1e-4 | 5.5e+07 |
| basic | 2 | 0.395 | 3e-4 | 6.11e+06 | 1e-4 | 5.5e+07 |
| basic | 4 | 0.415 | 3e-4 | 6.11e+06 | 3e-5 | 6.11e+08 |
| basic | 8 | 0.473 | 3e-5 | 6.11e+08 | 0 (stopped with no noise) | ∞ |
| cosine | 1 | 0.368 | 0.003 | 6.12e+04 | 0.001 | 5.5e+05 |
| cosine | 2 | 0.395 | 0.003 | 6.12e+04 | 0.001 | 5.5e+05 |
| cosine | 4 | 0.415 | 0.003 | 6.12e+04 | 0.001 | 5.5e+05 |
| cosine | 8 | 0.473 | 0.003 | 6.12e+04 | 0.001 | 5.5e+05 |

ε is for one Gaussian release (δ = 1e-5, sampling rate 1, one step). It is not the DP-SGD training ε.

## 3. Pulling the Two Effects Apart

Four attacks on the same images at the same σ:
- **Sum:** the attack on the noisy sum of K examples.
- **Informed:** an attacker who already knows the other K−1 examples. It subtracts their gradients and is left with one example plus the same noise. For a single step on a fresh model this is exactly the K = 1 attack at the same σ, so we reuse the K = 1 results.
- **Matched SNR:** one example attacked alone, with its noise shrunk so that its noise-to-signal ratio equals what that example has inside the K-sum (σ′ = σ·‖gᵢ‖ / ‖Σⱼ gⱼ‖). From `scripts/attack_matched_snr.py`.
- **Mixing only:** the K-sum with no noise.

Two numbers, in plain words:
- **Signal benefit = Matched SNR − Informed.** How much the bigger summed signal weakens the noise, for one example. Positive means the bigger sum helps the attacker.
- **Mixing cost = Matched SNR − Sum.** How much harder it is to pull one example out of a mix of K, at the same noise-to-signal ratio. Positive means mixing protects.
- **Net = Sum − Informed = Signal benefit − Mixing cost.** Positive means the signal effect wins; negative means mixing wins.

| Attacker | K | σ | Sum | Informed (K=1) | Matched SNR | Mixing only (σ=0) | Signal benefit | Mixing cost | Net |
|---|---|---|---|---|---|---|---|---|---|
| basic | 2 | 0 | 0.931 ± 0.090 | 0.965 ± 0.071 | 0.965 ± 0.071 | 0.931 ± 0.090 | +0.000 | +0.034 | -0.034 |
| basic | 2 | 3e-5 | 0.764 ± 0.098 | 0.858 ± 0.087 | — | 0.931 ± 0.090 | — | — | -0.094 |
| basic | 2 | 1e-4 | 0.528 ± 0.075 | 0.638 ± 0.080 | 0.769 ± 0.051 | 0.931 ± 0.090 | +0.131 | +0.242 | -0.110 |
| basic | 2 | 3e-4 | 0.355 ± 0.036 | 0.440 ± 0.038 | 0.536 ± 0.031 | 0.931 ± 0.090 | +0.096 | +0.181 | -0.085 |
| basic | 2 | 0.001 | 0.156 ± 0.044 | 0.269 ± 0.033 | 0.362 ± 0.019 | 0.931 ± 0.090 | +0.093 | +0.206 | -0.113 |
| basic | 2 | 0.003 | 0.045 ± 0.006 | 0.054 ± 0.006 | 0.134 ± 0.043 | 0.931 ± 0.090 | +0.080 | +0.089 | -0.009 |
| basic | 2 | 0.01 | 0.027 ± 0.003 | 0.030 ± 0.004 | — | 0.931 ± 0.090 | — | — | -0.003 |
| basic | 2 | 0.1 | 0.010 ± 0.005 | 0.005 ± 0.004 | — | 0.931 ± 0.090 | — | — | +0.005 |
| basic | 4 | 0 | 0.814 ± 0.139 | 0.965 ± 0.071 | 0.965 ± 0.071 | 0.814 ± 0.139 | +0.000 | +0.151 | -0.151 |
| basic | 4 | 3e-5 | 0.607 ± 0.113 | 0.858 ± 0.087 | — | 0.814 ± 0.139 | — | — | -0.251 |
| basic | 4 | 1e-4 | 0.395 ± 0.054 | 0.638 ± 0.080 | 0.886 ± 0.033 | 0.814 ± 0.139 | +0.247 | +0.490 | -0.243 |
| basic | 4 | 3e-4 | 0.234 ± 0.052 | 0.440 ± 0.038 | 0.658 ± 0.039 | 0.814 ± 0.139 | +0.219 | +0.424 | -0.206 |
| basic | 4 | 0.001 | 0.069 ± 0.012 | 0.269 ± 0.033 | 0.443 ± 0.015 | 0.814 ± 0.139 | +0.174 | +0.374 | -0.200 |
| basic | 4 | 0.003 | 0.038 ± 0.005 | 0.054 ± 0.006 | 0.274 ± 0.015 | 0.814 ± 0.139 | +0.220 | +0.236 | -0.016 |
| basic | 4 | 0.01 | 0.024 ± 0.003 | 0.030 ± 0.004 | — | 0.814 ± 0.139 | — | — | -0.006 |
| basic | 4 | 0.1 | 0.015 ± 0.004 | 0.005 ± 0.004 | — | 0.814 ± 0.139 | — | — | +0.010 |
| basic | 8 | 0 | 0.506 ± 0.052 | 0.965 ± 0.071 | 0.965 ± 0.071 | 0.506 ± 0.052 | +0.000 | +0.458 | -0.458 |
| basic | 8 | 3e-5 | 0.415 ± 0.037 | 0.858 ± 0.087 | — | 0.506 ± 0.052 | — | — | -0.443 |
| basic | 8 | 1e-4 | 0.265 ± 0.037 | 0.638 ± 0.080 | 0.934 ± 0.064 | 0.506 ± 0.052 | +0.296 | +0.669 | -0.373 |
| basic | 8 | 3e-4 | 0.109 ± 0.022 | 0.440 ± 0.038 | 0.802 ± 0.044 | 0.506 ± 0.052 | +0.363 | +0.693 | -0.331 |
| basic | 8 | 0.001 | 0.045 ± 0.006 | 0.269 ± 0.033 | 0.544 ± 0.026 | 0.506 ± 0.052 | +0.275 | +0.499 | -0.224 |
| basic | 8 | 0.003 | 0.037 ± 0.002 | 0.054 ± 0.006 | 0.386 ± 0.017 | 0.506 ± 0.052 | +0.332 | +0.349 | -0.017 |
| basic | 8 | 0.01 | 0.030 ± 0.003 | 0.030 ± 0.004 | — | 0.506 ± 0.052 | — | — | +0.000 |
| basic | 8 | 0.1 | 0.021 ± 0.002 | 0.005 ± 0.004 | — | 0.506 ± 0.052 | — | — | +0.016 |
| cosine | 2 | 0 | 0.735 ± 0.183 | 0.958 ± 0.053 | 0.958 ± 0.053 | 0.735 ± 0.183 | +0.000 | +0.223 | -0.223 |
| cosine | 2 | 3e-5 | 0.733 ± 0.178 | 0.935 ± 0.060 | — | 0.735 ± 0.183 | — | — | -0.202 |
| cosine | 2 | 1e-4 | 0.747 ± 0.160 | 0.901 ± 0.055 | 0.955 ± 0.044 | 0.735 ± 0.183 | +0.054 | +0.208 | -0.154 |
| cosine | 2 | 3e-4 | 0.687 ± 0.147 | 0.775 ± 0.065 | 0.883 ± 0.089 | 0.735 ± 0.183 | +0.108 | +0.196 | -0.088 |
| cosine | 2 | 0.001 | 0.433 ± 0.062 | 0.411 ± 0.022 | 0.573 ± 0.070 | 0.735 ± 0.183 | +0.162 | +0.140 | +0.022 |
| cosine | 2 | 0.003 | 0.197 ± 0.023 | 0.214 ± 0.022 | 0.310 ± 0.030 | 0.735 ± 0.183 | +0.096 | +0.113 | -0.017 |
| cosine | 2 | 0.01 | 0.059 ± 0.010 | 0.072 ± 0.011 | — | 0.735 ± 0.183 | — | — | -0.012 |
| cosine | 2 | 0.1 | 0.012 ± 0.005 | 0.007 ± 0.003 | — | 0.735 ± 0.183 | — | — | +0.005 |
| cosine | 4 | 0 | 0.607 ± 0.233 | 0.958 ± 0.053 | 0.958 ± 0.053 | 0.607 ± 0.233 | +0.000 | +0.350 | -0.350 |
| cosine | 4 | 3e-5 | 0.600 ± 0.193 | 0.935 ± 0.060 | — | 0.607 ± 0.233 | — | — | -0.335 |
| cosine | 4 | 1e-4 | 0.610 ± 0.199 | 0.901 ± 0.055 | 0.953 ± 0.050 | 0.607 ± 0.233 | +0.052 | +0.343 | -0.291 |
| cosine | 4 | 3e-4 | 0.599 ± 0.207 | 0.775 ± 0.065 | 0.946 ± 0.050 | 0.607 ± 0.233 | +0.171 | +0.347 | -0.176 |
| cosine | 4 | 0.001 | 0.445 ± 0.102 | 0.411 ± 0.022 | 0.782 ± 0.080 | 0.607 ± 0.233 | +0.372 | +0.338 | +0.034 |
| cosine | 4 | 0.003 | 0.198 ± 0.025 | 0.214 ± 0.022 | 0.428 ± 0.027 | 0.607 ± 0.233 | +0.214 | +0.230 | -0.016 |
| cosine | 4 | 0.01 | 0.057 ± 0.009 | 0.072 ± 0.011 | — | 0.607 ± 0.233 | — | — | -0.014 |
| cosine | 4 | 0.1 | 0.016 ± 0.001 | 0.007 ± 0.003 | — | 0.607 ± 0.233 | — | — | +0.009 |
| cosine | 8 | 0 | 0.467 ± 0.187 | 0.958 ± 0.053 | 0.958 ± 0.053 | 0.467 ± 0.187 | +0.000 | +0.491 | -0.491 |
| cosine | 8 | 3e-5 | 0.452 ± 0.210 | 0.935 ± 0.060 | — | 0.467 ± 0.187 | — | — | -0.483 |
| cosine | 8 | 1e-4 | 0.459 ± 0.207 | 0.901 ± 0.055 | 0.958 ± 0.041 | 0.467 ± 0.187 | +0.057 | +0.498 | -0.442 |
| cosine | 8 | 3e-4 | 0.455 ± 0.190 | 0.775 ± 0.065 | 0.953 ± 0.061 | 0.467 ± 0.187 | +0.178 | +0.498 | -0.320 |
| cosine | 8 | 0.001 | 0.397 ± 0.158 | 0.411 ± 0.022 | 0.904 ± 0.074 | 0.467 ± 0.187 | +0.493 | +0.507 | -0.014 |
| cosine | 8 | 0.003 | 0.223 ± 0.047 | 0.214 ± 0.022 | 0.614 ± 0.079 | 0.467 ± 0.187 | +0.400 | +0.391 | +0.009 |
| cosine | 8 | 0.01 | 0.062 ± 0.011 | 0.072 ± 0.011 | — | 0.467 ± 0.187 | — | — | -0.010 |
| cosine | 8 | 0.1 | 0.024 ± 0.002 | 0.007 ± 0.003 | — | 0.467 ± 0.187 | — | — | +0.017 |

Figure: `results/fig_rq4_decomposition.png` (K = 4 and 8).

## 4. Measured Success vs the Formal Bound

Reconstruction robustness (ReRo) bound for one release of the Gaussian mechanism (Hayes et al. 2023, arXiv:2302.07225, Corollary 4, with q = 1 and T = 1):

  success ≤ Φ(Φ⁻¹(κ) + 1/σ) = 1 − Φ(Φ⁻¹(1 − κ) − 1/σ)

- The paper gives the general bound as a blow-up function. The closed form above, for one full-batch step, is **our own reduction** (a Gaussian mean shift of 1/σ). Check it before citing.
- κ is the best success an attacker can get **without** seeing any gradient: the best single fixed guess, scored on random MNIST test digits at SSIM ≥ 0.9. We measured κ̂ = 0.007 over 1000 targets and 302 guesses (random training digits, the average digit, all black). Taking the best of many guesses makes κ̂ slightly too high, which only loosens the bound.
- The bound covers the informed attacker (who knows the other K−1 examples), so it covers every K. Measured success is shown for K = 1 (the informed attacker) and for the highest across all K.
- The bound is about any attacker. Our attacks are only two attackers, so measured success below the bound is expected; the gap shows how loose the bound is in practice, or how much better an attacker could do.

| σ | ε (one release) | ReRo bound | Basic, K=1 | Cosine, K=1 | Highest over K and attackers |
|---|---|---|---|---|---|
| 0 | ∞ | 100.0% | 94% | 92% | 94% |
| 3e-5 | 6.11e+08 | 100.0% | 76% | 89% | 89% |
| 1e-4 | 5.5e+07 | 100.0% | 0% | 82% | 82% |
| 3e-4 | 6.11e+06 | 100.0% | 0% | 26% | 26% |
| 0.001 | 5.5e+05 | 100.0% | 0% | 0% | 0% |
| 0.003 | 6.12e+04 | 100.0% | 0% | 0% | 0% |
| 0.01 | 5.61e+03 | 100.0% | 0% | 0% | 0% |
| 0.1 | 96.1 | 100.0% | 0% | 0% | 0% |
| 0.3 | 20.4 | 81.0% | — | — | not tested |
| 1 | 4.73 | 7.3% | — | — | not tested |
| 2 | 2.17 | 2.5% | — | — | not tested |

The bound first drops below 99% at σ = 0.3 (ε = 20.4).
Both attacks already fail (≤ 1% success at every K) from σ = 0.001 (ε = 5.5e+05), where the bound guarantees nothing.

