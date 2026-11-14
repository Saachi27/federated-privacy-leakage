# Leak From the Sum: MNIST, fresh LeNet, real FedAvg update (5 local steps at lr 0.01)

**Setting:** MNIST, fresh LeNet, real FedAvg update (5 local steps at lr 0.01). Test images 50-81, seeds [0, 1, 2]. The client trains on its K images and uploads the change in weights; the attacker treats it as a gradient. No noise is added in this setting, because noise on a whole client update is a different, client-level DP mechanism. Cosine attacker TV weight 0.003 (tuned on held-out sums, `tuning.json`). Attackers are given the true labels; images are scored against their best-matched original. SSIM is the standard form. The leak line is the **label-aware baseline** at the same K: the better of the class-average image and same-class real images, best-matched.

Every ± is a 95% t-interval over seeds of a difference **paired by seed** (the same seed fixes the model, the grouping of the images and the noise draw). Baselines have more seeds than the attacks; pairs use the shared seeds.

## 1. Leak Above the Label-Aware Baseline

Mean SSIM minus the baseline at the same K, paired by seed. **Bold** = clear leak (the whole interval is above 0). *Italic* = stopped (the whole interval is at or below 0). K = 16 was run without noise only.

**Basic (iDLG, L-BFGS)**

| K | σ=0 |
|---|---|
| 1 | -0.078 ± 0.130 |
| 2 | -0.120 ± 0.207 |
| 4 | -0.137 ± 0.143 |
| 8 | *-0.209 ± 0.107* |
| 16 | *-0.262 ± 0.039* |

**Cosine (Geiping et al.)**

| K | σ=0 |
|---|---|
| 1 | **+0.384 ± 0.097** |
| 2 | +0.136 ± 0.245 |
| 4 | +0.036 ± 0.249 |
| 8 | -0.059 ± 0.218 |
| 16 | *-0.155 ± 0.099* |

**Without noise, the same test with PSNR as a cross-check** (attack minus label-aware PSNR baseline, dB, capped at 60):

| Attacker | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|
| basic | *-6.4 ± 1.5* | *-8.2 ± 3.0* | *-8.3 ± 2.5* | *-9.0 ± 0.9* | *-9.2 ± 0.4* |
| cosine | +3.4 ± 3.7 | -3.4 ± 4.3 | *-4.7 ± 4.1* | *-5.3 ± 3.5* | *-5.5 ± 2.7* |

No noise was tested in this setting, so there is no stopping noise, informed-attacker comparison or reconstruction bound here.

