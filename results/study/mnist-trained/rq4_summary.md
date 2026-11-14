# Noise on the Sum and What Mixing Protects: MNIST, LeNet trained with FedAvg for 20 rounds, one FedSGD step

**Setting:** MNIST, LeNet trained with FedAvg for 20 rounds, one FedSGD step. Test images 50-81, seeds [0, 1, 2]. Trained model test accuracy: seed 0: 0.937, seed 1: 0.938, seed 2: 0.936. Gaussian noise σ·C is added once to the sum of K clipped per-example gradients (clip C = 27.1). Cosine attacker TV weight 0.01 (tuned on held-out sums, `tuning.json`). Attackers are given the true labels; images are scored against their best-matched original. SSIM is the standard form. The leak line is the **label-aware baseline** at the same K: the better of the class-average image and same-class real images, best-matched.

Every ± is a 95% t-interval over seeds of a difference **paired by seed** (the same seed fixes the model, the grouping of the images and the noise draw). Baselines have more seeds than the attacks; pairs use the shared seeds.

## 1. Leak Above the Label-Aware Baseline

Mean SSIM minus the baseline at the same K, paired by seed. **Bold** = clear leak (the whole interval is above 0). *Italic* = stopped (the whole interval is at or below 0). K = 16 was run without noise only.

**Basic (iDLG, L-BFGS)**

| K | σ=0 | σ=1e-4 | σ=3e-4 | σ=0.001 | σ=0.003 | σ=0.01 | σ=0.03 | σ=0.1 |
|---|---|---|---|---|---|---|---|---|
| 1 | -0.160 ± 0.257 | *-0.230 ± 0.137* | *-0.215 ± 0.157* | *-0.242 ± 0.055* | *-0.258 ± 0.011* | *-0.265 ± 0.030* | *-0.263 ± 0.010* | *-0.258 ± 0.017* |
| 2 | -0.147 ± 0.212 | *-0.173 ± 0.046* | *-0.204 ± 0.034* | *-0.221 ± 0.036* | *-0.231 ± 0.014* | *-0.236 ± 0.006* | *-0.236 ± 0.024* | *-0.237 ± 0.020* |
| 4 | +0.021 ± 0.213 | -0.079 ± 0.142 | *-0.161 ± 0.093* | *-0.193 ± 0.054* | *-0.216 ± 0.014* | *-0.224 ± 0.016* | *-0.224 ± 0.010* | *-0.227 ± 0.029* |
| 8 | -0.024 ± 0.077 | -0.019 ± 0.138 | *-0.149 ± 0.079* | *-0.214 ± 0.091* | *-0.244 ± 0.085* | *-0.246 ± 0.064* | *-0.250 ± 0.069* | *-0.245 ± 0.074* |
| 16 | *-0.140 ± 0.056* | — | — | — | — | — | — | — |

**Cosine (Geiping et al.)**

| K | σ=0 | σ=1e-4 | σ=3e-4 | σ=0.001 | σ=0.003 | σ=0.01 | σ=0.03 | σ=0.1 |
|---|---|---|---|---|---|---|---|---|
| 1 | **+0.686 ± 0.024** | **+0.418 ± 0.108** | **+0.237 ± 0.108** | +0.049 ± 0.109 | *-0.097 ± 0.086* | *-0.197 ± 0.056* | *-0.238 ± 0.021* | *-0.251 ± 0.023* |
| 2 | **+0.352 ± 0.072** | **+0.297 ± 0.079** | **+0.187 ± 0.074** | +0.027 ± 0.058 | *-0.107 ± 0.046* | *-0.199 ± 0.024* | *-0.220 ± 0.012* | *-0.224 ± 0.012* |
| 4 | **+0.173 ± 0.058** | **+0.155 ± 0.115** | +0.108 ± 0.127 | +0.008 ± 0.063 | *-0.114 ± 0.014* | *-0.189 ± 0.016* | *-0.212 ± 0.008* | *-0.212 ± 0.014* |
| 8 | -0.015 ± 0.031 | -0.011 ± 0.039 | -0.029 ± 0.044 | -0.079 ± 0.088 | *-0.145 ± 0.077* | *-0.221 ± 0.056* | *-0.232 ± 0.068* | *-0.237 ± 0.065* |
| 16 | *-0.127 ± 0.070* | — | — | — | — | — | — | — |

**Without noise, the same test with PSNR as a cross-check** (attack minus label-aware PSNR baseline, dB, capped at 60):

| Attacker | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|
| basic | -5.7 ± 10.1 | -6.4 ± 6.8 | -3.0 ± 5.6 | *-4.2 ± 3.7* | *-6.3 ± 2.0* |
| cosine | **+15.2 ± 1.9** | **+5.2 ± 3.5** | +0.8 ± 2.1 | *-3.2 ± 1.4* | *-4.9 ± 2.9* |

## 2. Stopping Noise σ*(K)

The lowest tested σ from which the attack is stopped at that σ and at every larger tested σ. "Not stopped" means no tested σ stops it. Three rules:
- **SSIM rule (main):** the paired 95% interval of attack minus label-aware baseline is at or below 0.
- **PSNR rule (cross-check):** the same test with PSNR (capped at 60 dB) against the label-aware PSNR baseline.
- **Success rule:** at most 1% of images reach SSIM ≥ 0.9. The 0.9 line misses noisy but readable images, so this rule stops too early; it is shown for comparison only.

| Attacker | K | Baseline SSIM | σ* (SSIM) | ε at σ* | σ* (PSNR) | ε at σ* | σ* (success) | ε at σ* |
|---|---|---|---|---|---|---|---|---|
| basic | 1 | 0.266 | 1e-4 | 5.5e+07 | 1e-4 | 5.5e+07 | 3e-4 | 6.11e+06 |
| basic | 2 | 0.266 | 1e-4 | 5.5e+07 | 1e-4 | 5.5e+07 | 3e-4 | 6.11e+06 |
| basic | 4 | 0.273 | 3e-4 | 6.11e+06 | 1e-4 | 5.5e+07 | 3e-4 | 6.11e+06 |
| basic | 8 | 0.302 | 3e-4 | 6.11e+06 | 0 (stopped with no noise) | ∞ | 1e-4 | 5.5e+07 |
| cosine | 1 | 0.266 | 0.003 | 6.12e+04 | 0.001 | 5.5e+05 | 0.003 | 6.12e+04 |
| cosine | 2 | 0.266 | 0.003 | 6.12e+04 | 0.001 | 5.5e+05 | 0.001 | 5.5e+05 |
| cosine | 4 | 0.273 | 0.003 | 6.12e+04 | 0.001 | 5.5e+05 | 0.001 | 5.5e+05 |
| cosine | 8 | 0.302 | 0.003 | 6.12e+04 | 0 (stopped with no noise) | ∞ | 3e-4 | 6.11e+06 |

ε is for one Gaussian release (δ = 1e-5, sampling rate 1, one step, add/remove-one sensitivity C). It is not the DP-SGD training ε.

## 3. The Sum vs an Attacker Who Knows the Other Examples

- **Sum:** the attack on the noisy sum of K examples.
- **Informed:** an attacker who already knows the other K−1 examples. It subtracts their gradients and is left with one example plus noise of the same size. For a single step this is **equal in distribution** to the K = 1 attack at the same σ (same noise size, not the same noise draw), so the K = 1 results stand in for it.

**What to expect before looking.** The informed attacker has at least as much information as the attacker who sees only the sum: it can always compute the sum itself. So for an optimal attacker the net effect (Sum − Informed) cannot be positive. A negative net is therefore not a discovery; the useful result is **how large the gap is** for these two practical attackers, and at which noise levels it closes.

| Attacker | K | σ | Sum | Informed (K=1) | Net (Sum − Informed) |
|---|---|---|---|---|---|
| basic | 2 | 0 | 0.119 ± 0.212 | 0.106 ± 0.257 | +0.013 ± 0.057 |
| basic | 2 | 1e-4 | 0.093 ± 0.046 | 0.036 ± 0.137 | +0.057 ± 0.110 |
| basic | 2 | 3e-4 | 0.063 ± 0.034 | 0.052 ± 0.157 | +0.011 ± 0.129 |
| basic | 2 | 0.001 | 0.045 ± 0.036 | 0.025 ± 0.055 | **+0.020 ± 0.019** |
| basic | 2 | 0.003 | 0.036 ± 0.014 | 0.008 ± 0.011 | **+0.027 ± 0.026** |
| basic | 2 | 0.01 | 0.030 ± 0.006 | 0.002 ± 0.030 | **+0.028 ± 0.024** |
| basic | 2 | 0.03 | 0.031 ± 0.024 | 0.003 ± 0.010 | **+0.028 ± 0.014** |
| basic | 2 | 0.1 | 0.030 ± 0.020 | 0.008 ± 0.017 | +0.021 ± 0.038 |
| basic | 4 | 0 | 0.288 ± 0.213 | 0.106 ± 0.257 | +0.182 ± 0.363 |
| basic | 4 | 1e-4 | 0.188 ± 0.142 | 0.036 ± 0.137 | **+0.151 ± 0.136** |
| basic | 4 | 3e-4 | 0.105 ± 0.093 | 0.052 ± 0.157 | +0.054 ± 0.198 |
| basic | 4 | 0.001 | 0.073 ± 0.054 | 0.025 ± 0.055 | +0.048 ± 0.080 |
| basic | 4 | 0.003 | 0.051 ± 0.014 | 0.008 ± 0.011 | **+0.042 ± 0.009** |
| basic | 4 | 0.01 | 0.043 ± 0.016 | 0.002 ± 0.030 | **+0.041 ± 0.036** |
| basic | 4 | 0.03 | 0.042 ± 0.010 | 0.003 ± 0.010 | **+0.039 ± 0.014** |
| basic | 4 | 0.1 | 0.039 ± 0.029 | 0.008 ± 0.017 | **+0.031 ± 0.011** |
| basic | 8 | 0 | 0.272 ± 0.117 | 0.106 ± 0.257 | **+0.165 ± 0.140** |
| basic | 8 | 1e-4 | 0.277 ± 0.100 | 0.036 ± 0.137 | **+0.241 ± 0.152** |
| basic | 8 | 3e-4 | 0.147 ± 0.011 | 0.052 ± 0.157 | +0.095 ± 0.166 |
| basic | 8 | 0.001 | 0.083 ± 0.021 | 0.025 ± 0.055 | +0.058 ± 0.076 |
| basic | 8 | 0.003 | 0.052 ± 0.019 | 0.008 ± 0.011 | **+0.044 ± 0.026** |
| basic | 8 | 0.01 | 0.050 ± 0.009 | 0.002 ± 0.030 | **+0.049 ± 0.022** |
| basic | 8 | 0.03 | 0.046 ± 0.007 | 0.003 ± 0.010 | **+0.043 ± 0.003** |
| basic | 8 | 0.1 | 0.051 ± 0.007 | 0.008 ± 0.017 | **+0.043 ± 0.011** |
| cosine | 2 | 0 | 0.618 ± 0.072 | 0.952 ± 0.024 | **-0.334 ± 0.091** |
| cosine | 2 | 1e-4 | 0.563 ± 0.079 | 0.684 ± 0.108 | **-0.121 ± 0.105** |
| cosine | 2 | 3e-4 | 0.454 ± 0.074 | 0.504 ± 0.108 | **-0.050 ± 0.047** |
| cosine | 2 | 0.001 | 0.294 ± 0.058 | 0.316 ± 0.109 | -0.022 ± 0.051 |
| cosine | 2 | 0.003 | 0.159 ± 0.046 | 0.170 ± 0.086 | -0.010 ± 0.045 |
| cosine | 2 | 0.01 | 0.068 ± 0.024 | 0.069 ± 0.056 | -0.002 ± 0.034 |
| cosine | 2 | 0.03 | 0.046 ± 0.012 | 0.029 ± 0.021 | +0.017 ± 0.025 |
| cosine | 2 | 0.1 | 0.042 ± 0.012 | 0.015 ± 0.023 | +0.027 ± 0.028 |
| cosine | 4 | 0 | 0.439 ± 0.058 | 0.952 ± 0.024 | **-0.513 ± 0.082** |
| cosine | 4 | 1e-4 | 0.421 ± 0.115 | 0.684 ± 0.108 | **-0.263 ± 0.135** |
| cosine | 4 | 3e-4 | 0.375 ± 0.127 | 0.504 ± 0.108 | **-0.129 ± 0.091** |
| cosine | 4 | 0.001 | 0.275 ± 0.063 | 0.316 ± 0.109 | -0.041 ± 0.081 |
| cosine | 4 | 0.003 | 0.152 ± 0.014 | 0.170 ± 0.086 | -0.017 ± 0.084 |
| cosine | 4 | 0.01 | 0.077 ± 0.016 | 0.069 ± 0.056 | +0.008 ± 0.042 |
| cosine | 4 | 0.03 | 0.055 ± 0.008 | 0.029 ± 0.021 | +0.026 ± 0.028 |
| cosine | 4 | 0.1 | 0.055 ± 0.014 | 0.015 ± 0.023 | **+0.040 ± 0.037** |
| cosine | 8 | 0 | 0.281 ± 0.047 | 0.952 ± 0.024 | **-0.671 ± 0.067** |
| cosine | 8 | 1e-4 | 0.285 ± 0.043 | 0.684 ± 0.108 | **-0.399 ± 0.148** |
| cosine | 8 | 3e-4 | 0.267 ± 0.049 | 0.504 ± 0.108 | **-0.237 ± 0.154** |
| cosine | 8 | 0.001 | 0.217 ± 0.031 | 0.316 ± 0.109 | **-0.098 ± 0.078** |
| cosine | 8 | 0.003 | 0.151 ± 0.030 | 0.170 ± 0.086 | -0.019 ± 0.057 |
| cosine | 8 | 0.01 | 0.075 ± 0.016 | 0.069 ± 0.056 | +0.006 ± 0.062 |
| cosine | 8 | 0.03 | 0.064 ± 0.006 | 0.029 ± 0.021 | **+0.035 ± 0.026** |
| cosine | 8 | 0.1 | 0.059 ± 0.009 | 0.015 ± 0.023 | **+0.044 ± 0.031** |

**Bold** net = its 95% interval excludes 0. Where the interval includes 0 the data cannot tell the sum and the informed attacker apart; that is not evidence that they are equal.

## 4. Measured Success vs an Indicative Reconstruction Bound

Reconstruction-robustness (ReRo) bound for one release of the Gaussian mechanism, in the closed form of Kaissis et al. 2023 (arXiv:2307.03928), which agrees with Hayes et al. 2023 (arXiv:2302.07225, Corollary 4) at sampling rate 1 and one step:

  success ≤ Φ(Φ⁻¹(κ) + 1/σ) = 1 − Φ(Φ⁻¹(1 − κ) − 1/σ)

**This bound is indicative, not certified.**
- κ is the best success an attacker can reach **without** the gradient. The attackers know the labels, so κ is label-aware: for each class, the best single fixed guess among the class average and 50 same-class training images, scored on same-class mnist test images at standard SSIM ≥ 0.9, then weighted by class frequency. κ̂ = 0.0034.
- κ̂ is a **lower estimate** of the true κ: it searches finitely many candidate guesses, and a better guess may exist. A larger κ gives a larger (weaker) bound, so the true bound may be weaker than shown. The sampling error of κ̂ is not bounded either.
- The bound is about any attacker, including the informed one, so it covers every K. Our two attackers are far from optimal, so measured success below the bound is expected.

| σ | ε (one release) | Indicative bound | Basic, K=1 | Cosine, K=1 | Highest over K and attackers |
|---|---|---|---|---|---|
| 0 | ∞ | 100.0% | 8% | 83% | 83% |
| 1e-4 | 5.5e+07 | 100.0% | 1% | 29% | 29% |
| 3e-4 | 6.11e+06 | 100.0% | 0% | 19% | 19% |
| 0.001 | 5.5e+05 | 100.0% | 0% | 5% | 5% |
| 0.003 | 6.12e+04 | 100.0% | 0% | 0% | 0% |
| 0.01 | 5.61e+03 | 100.0% | 0% | 0% | 0% |
| 0.03 | 722 | 100.0% | 0% | 0% | 0% |
| 0.1 | 96.1 | 100.0% | 0% | 0% | 0% |
| 0.3 | 20.4 | 73.5% | — | — | not tested |
| 1 | 4.73 | 4.4% | — | — | not tested |
| 2 | 2.17 | 1.4% | — | — | not tested |

The indicative bound first drops below 99% at σ = 0.3 (ε = 20.4).
By the SSIM rule, both attacks are stopped at every K from σ = 0.003 (ε = 6.12e+04).

