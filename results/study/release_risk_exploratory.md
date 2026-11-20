# Risk per Release on the Saved Study Reconstructions (EXPLORATORY)

> **EXPLORATORY.** This reuses the old target images (MNIST/CIFAR-10 test 50–81), the old grouping and the old attack runs (seeds 0–2, no restarts). It is **not** the locked held-out design. Use it to see how the measures behave, not as a result. Made by `scripts/score_releases.py`.

## How to read this

- A **release** is one attacked upload: one group of K images (seed, K, group). No noise.
- **Mean SSIM:** mean valid-window SSIM of the K matched pairs.
- **Any image passes:** 1 if any matched reconstruction has SSIM ≥ 0.9. The table shows the rate.
- **Retrieval hit:** for each reconstruction, rank *all* test images of its class by SSIM and take the top-1, without knowing which images are in the release. A hit if any pick is one of the release's images. The table shows the hit rate.
- **Baseline (label-aware):** for the same release, the better (per measure) of two guesses that know the labels: the class-average training image, and a random other test image of the same class (best-matched). These are the guesses of `scripts/baselines.py`, regenerated with the same draws.
- Every interval is a 95% interval over releases, **stratified by checkpoint** (seed). The main one is a stratified t-interval; the paired bootstrap (10,000 resamples, seed 0) is in brackets. **Bold** = the attack beats the baseline (lower end of the paired 95% t-interval of attack − baseline above 0).
- Releases per cell: K = 1: 96, K = 2: 48, K = 4: 24, K = 8: 12, K = 16: 6 (two per seed, so its intervals are rough).

## MNIST, fresh LeNet, one FedSGD step (`mnist-fresh`)

**Mean SSIM**

| K | Attacker | Releases | Attack | Baseline | Attack − baseline |
|---|---|---|---|---|---|
| 1 | basic | 96 | 0.995 [0.991, 1.000] | 0.316 [0.284, 0.349] | **+0.679 [0.647, 0.711] (boot [0.648, 0.710])** |
| 1 | cosine | 96 | 0.985 [0.980, 0.991] | 0.316 [0.284, 0.349] | **+0.669 [0.637, 0.700] (boot [0.638, 0.699])** |
| 2 | basic | 48 | 0.965 [0.934, 0.997] | 0.294 [0.262, 0.326] | **+0.671 [0.628, 0.715] (boot [0.631, 0.710])** |
| 2 | cosine | 48 | 0.858 [0.809, 0.906] | 0.294 [0.262, 0.326] | **+0.563 [0.510, 0.617] (boot [0.512, 0.612])** |
| 4 | basic | 24 | 0.936 [0.897, 0.974] | 0.290 [0.266, 0.314] | **+0.645 [0.604, 0.687] (boot [0.611, 0.680])** |
| 4 | cosine | 24 | 0.729 [0.660, 0.797] | 0.290 [0.266, 0.314] | **+0.438 [0.370, 0.507] (boot [0.379, 0.499])** |
| 8 | basic | 12 | 0.784 [0.727, 0.840] | 0.307 [0.282, 0.333] | **+0.477 [0.427, 0.526] (boot [0.441, 0.513])** |
| 8 | cosine | 12 | 0.561 [0.495, 0.627] | 0.307 [0.282, 0.333] | **+0.254 [0.194, 0.313] (boot [0.211, 0.295])** |
| 16 | basic | 6 | 0.269 [0.219, 0.319] | 0.333 [0.262, 0.405] | -0.064 [-0.112, -0.016] (boot [-0.080, -0.049]) |
| 16 | cosine | 6 | 0.369 [0.313, 0.425] | 0.333 [0.262, 0.405] | +0.036 [-0.048, 0.120] (boot [0.015, 0.056]) |

**Any image passes (SSIM ≥ 0.9)**

| K | Attacker | Releases | Attack | Baseline | Attack − baseline |
|---|---|---|---|---|---|
| 1 | basic | 96 | 0.969 [0.933, 1.004] | 0.000 [0.000, 0.000] | **+0.969 [0.933, 1.004] (boot [0.927, 1.000])** |
| 1 | cosine | 96 | 0.958 [0.918, 0.999] | 0.000 [0.000, 0.000] | **+0.958 [0.918, 0.999] (boot [0.917, 0.990])** |
| 2 | basic | 48 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |
| 2 | cosine | 48 | 0.854 [0.754, 0.955] | 0.000 [0.000, 0.000] | **+0.854 [0.754, 0.955] (boot [0.750, 0.938])** |
| 4 | basic | 24 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |
| 4 | cosine | 24 | 0.625 [0.425, 0.825] | 0.000 [0.000, 0.000] | **+0.625 [0.425, 0.825] (boot [0.458, 0.792])** |
| 8 | basic | 12 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |
| 8 | cosine | 12 | 0.333 [-0.012, 0.679] | 0.000 [0.000, 0.000] | +0.333 [-0.012, 0.679] (boot [0.083, 0.583]) |
| 16 | basic | 6 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] (boot [0.000, 0.000]) |
| 16 | cosine | 6 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] (boot [0.000, 0.000]) |

**Retrieval hit (top-1 in class)**

| K | Attacker | Releases | Attack | Baseline | Attack − baseline |
|---|---|---|---|---|---|
| 1 | basic | 96 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |
| 1 | cosine | 96 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |
| 2 | basic | 48 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |
| 2 | cosine | 48 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |
| 4 | basic | 24 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |
| 4 | cosine | 24 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |
| 8 | basic | 12 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |
| 8 | cosine | 12 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |
| 16 | basic | 6 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |
| 16 | cosine | 6 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |

**Share of single reconstructions whose top-1 pick is exactly their own target** (per image, attack only; shows that low-SSIM reconstructions can still identify the image):

| Attacker | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|
| basic | 100% | 99% | 96% | 95% | 57% |
| cosine | 100% | 96% | 84% | 73% | 44% |

Retrieval by chance (one uniform random pick per target in its class): K = 1: 0.10%, K = 2: 0.22%, K = 4: 0.50%, K = 8: 1.26%, K = 16: 3.71%.
Check: recomputed SSIM of the saved reconstructions differs from the saved records by at most 7.2e-05 (float16 storage).

## MNIST, trained LeNet, one FedSGD step (`mnist-trained`)

**Mean SSIM**

| K | Attacker | Releases | Attack | Baseline | Attack − baseline |
|---|---|---|---|---|---|
| 1 | basic | 96 | 0.106 [0.048, 0.164] | 0.316 [0.284, 0.349] | -0.210 [-0.270, -0.150] (boot [-0.267, -0.150]) |
| 1 | cosine | 96 | 0.952 [0.939, 0.966] | 0.316 [0.284, 0.349] | **+0.636 [0.602, 0.669] (boot [0.603, 0.668])** |
| 2 | basic | 48 | 0.119 [0.048, 0.191] | 0.294 [0.262, 0.326] | -0.175 [-0.253, -0.096] (boot [-0.244, -0.097]) |
| 2 | cosine | 48 | 0.618 [0.580, 0.656] | 0.294 [0.262, 0.326] | **+0.324 [0.271, 0.376] (boot [0.274, 0.372])** |
| 4 | basic | 24 | 0.288 [0.182, 0.394] | 0.290 [0.266, 0.314] | -0.002 [-0.100, 0.095] (boot [-0.087, 0.084]) |
| 4 | cosine | 24 | 0.439 [0.408, 0.470] | 0.290 [0.266, 0.314] | **+0.149 [0.107, 0.191] (boot [0.112, 0.185])** |
| 8 | basic | 12 | 0.272 [0.164, 0.379] | 0.307 [0.282, 0.333] | -0.036 [-0.152, 0.080] (boot [-0.123, 0.050]) |
| 8 | cosine | 12 | 0.281 [0.246, 0.315] | 0.307 [0.282, 0.333] | -0.026 [-0.063, 0.010] (boot [-0.052, -0.001]) |
| 16 | basic | 6 | 0.194 [0.038, 0.350] | 0.333 [0.262, 0.405] | -0.140 [-0.284, 0.004] (boot [-0.187, -0.092]) |
| 16 | cosine | 6 | 0.207 [0.144, 0.269] | 0.333 [0.262, 0.405] | -0.127 [-0.183, -0.070] (boot [-0.147, -0.106]) |

**Any image passes (SSIM ≥ 0.9)**

| K | Attacker | Releases | Attack | Baseline | Attack − baseline |
|---|---|---|---|---|---|
| 1 | basic | 96 | 0.083 [0.028, 0.139] | 0.000 [0.000, 0.000] | **+0.083 [0.028, 0.139] (boot [0.031, 0.135])** |
| 1 | cosine | 96 | 0.833 [0.758, 0.908] | 0.000 [0.000, 0.000] | **+0.833 [0.758, 0.908] (boot [0.760, 0.906])** |
| 2 | basic | 48 | 0.104 [0.014, 0.195] | 0.000 [0.000, 0.000] | **+0.104 [0.014, 0.195] (boot [0.021, 0.188])** |
| 2 | cosine | 48 | 0.479 [0.334, 0.624] | 0.000 [0.000, 0.000] | **+0.479 [0.334, 0.624] (boot [0.333, 0.625])** |
| 4 | basic | 24 | 0.333 [0.121, 0.546] | 0.000 [0.000, 0.000] | **+0.333 [0.121, 0.546] (boot [0.167, 0.542])** |
| 4 | cosine | 24 | 0.292 [0.087, 0.496] | 0.000 [0.000, 0.000] | **+0.292 [0.087, 0.496] (boot [0.125, 0.458])** |
| 8 | basic | 12 | 0.167 [-0.122, 0.455] | 0.000 [0.000, 0.000] | +0.167 [-0.122, 0.455] (boot [0.000, 0.333]) |
| 8 | cosine | 12 | 0.083 [-0.182, 0.349] | 0.000 [0.000, 0.000] | +0.083 [-0.182, 0.349] (boot [0.000, 0.250]) |
| 16 | basic | 6 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] (boot [0.000, 0.000]) |
| 16 | cosine | 6 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] (boot [0.000, 0.000]) |

**Retrieval hit (top-1 in class)**

| K | Attacker | Releases | Attack | Baseline | Attack − baseline |
|---|---|---|---|---|---|
| 1 | basic | 96 | 0.104 [0.043, 0.165] | 0.000 [0.000, 0.000] | **+0.104 [0.043, 0.165] (boot [0.052, 0.167])** |
| 1 | cosine | 96 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |
| 2 | basic | 48 | 0.167 [0.059, 0.275] | 0.000 [0.000, 0.000] | **+0.167 [0.059, 0.275] (boot [0.062, 0.271])** |
| 2 | cosine | 48 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |
| 4 | basic | 24 | 0.667 [0.464, 0.870] | 0.000 [0.000, 0.000] | **+0.667 [0.464, 0.870] (boot [0.500, 0.833])** |
| 4 | cosine | 24 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |
| 8 | basic | 12 | 0.833 [0.545, 1.122] | 0.000 [0.000, 0.000] | **+0.833 [0.545, 1.122] (boot [0.583, 1.000])** |
| 8 | cosine | 12 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |
| 16 | basic | 6 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |
| 16 | cosine | 6 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |

**Share of single reconstructions whose top-1 pick is exactly their own target** (per image, attack only; shows that low-SSIM reconstructions can still identify the image):

| Attacker | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|
| basic | 10% | 12% | 36% | 38% | 29% |
| cosine | 100% | 80% | 55% | 31% | 26% |

Retrieval by chance (one uniform random pick per target in its class): K = 1: 0.10%, K = 2: 0.22%, K = 4: 0.50%, K = 8: 1.26%, K = 16: 3.71%.
Check: recomputed SSIM of the saved reconstructions differs from the saved records by at most 5.3e-05 (float16 storage).

## CIFAR-10, fresh LeNet, one FedSGD step (`cifar10-fresh`)

**Mean SSIM**

| K | Attacker | Releases | Attack | Baseline | Attack − baseline |
|---|---|---|---|---|---|
| 1 | basic | 96 | 0.626 [0.585, 0.666] | 0.164 [0.142, 0.186] | **+0.462 [0.417, 0.506] (boot [0.415, 0.501])** |
| 1 | cosine | 96 | 0.773 [0.759, 0.787] | 0.164 [0.142, 0.186] | **+0.609 [0.589, 0.628] (boot [0.590, 0.627])** |
| 2 | basic | 48 | 0.134 [0.094, 0.175] | 0.159 [0.141, 0.178] | -0.025 [-0.068, 0.018] (boot [-0.064, 0.016]) |
| 2 | cosine | 48 | 0.567 [0.528, 0.607] | 0.159 [0.141, 0.178] | **+0.408 [0.361, 0.455] (boot [0.365, 0.451])** |
| 4 | basic | 24 | 0.036 [0.033, 0.039] | 0.159 [0.140, 0.179] | -0.123 [-0.145, -0.102] (boot [-0.142, -0.105]) |
| 4 | cosine | 24 | 0.408 [0.367, 0.449] | 0.159 [0.140, 0.179] | **+0.249 [0.205, 0.293] (boot [0.211, 0.289])** |
| 8 | basic | 12 | 0.030 [0.026, 0.034] | 0.166 [0.147, 0.184] | -0.136 [-0.158, -0.114] (boot [-0.152, -0.121]) |
| 8 | cosine | 12 | 0.292 [0.259, 0.325] | 0.166 [0.147, 0.184] | **+0.126 [0.094, 0.159] (boot [0.104, 0.150])** |
| 16 | basic | 6 | 0.025 [0.024, 0.027] | 0.168 [0.122, 0.215] | -0.143 [-0.182, -0.104] (boot [-0.148, -0.138]) |
| 16 | cosine | 6 | 0.232 [0.197, 0.266] | 0.168 [0.122, 0.215] | **+0.063 [0.023, 0.103] (boot [0.053, 0.073])** |

**Any image passes (SSIM ≥ 0.9)**

| K | Attacker | Releases | Attack | Baseline | Attack − baseline |
|---|---|---|---|---|---|
| 1 | basic | 96 | 0.604 [0.555, 0.653] | 0.000 [0.000, 0.000] | **+0.604 [0.555, 0.653] (boot [0.552, 0.646])** |
| 1 | cosine | 96 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] (boot [0.000, 0.000]) |
| 2 | basic | 48 | 0.083 [0.004, 0.163] | 0.000 [0.000, 0.000] | **+0.083 [0.004, 0.163] (boot [0.021, 0.167])** |
| 2 | cosine | 48 | 0.021 [-0.024, 0.065] | 0.000 [0.000, 0.000] | +0.021 [-0.024, 0.065] (boot [0.000, 0.062]) |
| 4 | basic | 24 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] (boot [0.000, 0.000]) |
| 4 | cosine | 24 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] (boot [0.000, 0.000]) |
| 8 | basic | 12 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] (boot [0.000, 0.000]) |
| 8 | cosine | 12 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] (boot [0.000, 0.000]) |
| 16 | basic | 6 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] (boot [0.000, 0.000]) |
| 16 | cosine | 6 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] (boot [0.000, 0.000]) |

**Retrieval hit (top-1 in class)**

| K | Attacker | Releases | Attack | Baseline | Attack − baseline |
|---|---|---|---|---|---|
| 1 | basic | 96 | 0.667 [0.611, 0.723] | 0.000 [0.000, 0.000] | **+0.667 [0.611, 0.723] (boot [0.615, 0.719])** |
| 1 | cosine | 96 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |
| 2 | basic | 48 | 0.979 [0.935, 1.024] | 0.000 [0.000, 0.000] | **+0.979 [0.935, 1.024] (boot [0.938, 1.000])** |
| 2 | cosine | 48 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |
| 4 | basic | 24 | 0.833 [0.673, 0.994] | 0.000 [0.000, 0.000] | **+0.833 [0.673, 0.994] (boot [0.708, 0.958])** |
| 4 | cosine | 24 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |
| 8 | basic | 12 | 0.750 [0.437, 1.063] | 0.000 [0.000, 0.000] | **+0.750 [0.437, 1.063] (boot [0.500, 0.917])** |
| 8 | cosine | 12 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |
| 16 | basic | 6 | 0.500 [-0.419, 1.419] | 0.000 [0.000, 0.000] | +0.500 [-0.419, 1.419] (boot [0.167, 0.833]) |
| 16 | cosine | 6 | 0.833 [-1.284, 2.951] | 0.000 [0.000, 0.000] | +0.833 [-1.284, 2.951] (boot [0.667, 1.000]) |

**Share of single reconstructions whose top-1 pick is exactly their own target** (per image, attack only; shows that low-SSIM reconstructions can still identify the image):

| Attacker | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|
| basic | 67% | 76% | 36% | 14% | 5% |
| cosine | 100% | 86% | 75% | 48% | 19% |

Retrieval by chance (one uniform random pick per target in its class): K = 1: 0.10%, K = 2: 0.21%, K = 4: 0.52%, K = 8: 1.34%, K = 16: 3.96%.
Check: recomputed SSIM of the saved reconstructions differs from the saved records by at most 1.4e-04 (float16 storage).

## MNIST, fresh LeNet, FedAvg update (5 local steps, one client) (`mnist-fedavg`)

**Mean SSIM**

| K | Attacker | Releases | Attack | Baseline | Attack − baseline |
|---|---|---|---|---|---|
| 1 | basic | 96 | 0.189 [0.132, 0.245] | 0.316 [0.284, 0.349] | -0.128 [-0.194, -0.061] (boot [-0.189, -0.064]) |
| 1 | cosine | 96 | 0.650 [0.604, 0.696] | 0.316 [0.284, 0.349] | **+0.334 [0.274, 0.394] (boot [0.274, 0.390])** |
| 2 | basic | 48 | 0.147 [0.115, 0.178] | 0.294 [0.262, 0.326] | -0.148 [-0.194, -0.101] (boot [-0.191, -0.104]) |
| 2 | cosine | 48 | 0.403 [0.358, 0.447] | 0.294 [0.262, 0.326] | **+0.108 [0.056, 0.161] (boot [0.057, 0.157])** |
| 4 | basic | 24 | 0.129 [0.096, 0.162] | 0.290 [0.266, 0.314] | -0.161 [-0.207, -0.115] (boot [-0.200, -0.124]) |
| 4 | cosine | 24 | 0.302 [0.264, 0.341] | 0.290 [0.266, 0.314] | +0.012 [-0.035, 0.059] (boot [-0.029, 0.053]) |
| 8 | basic | 12 | 0.087 [0.077, 0.096] | 0.307 [0.282, 0.333] | -0.221 [-0.249, -0.192] (boot [-0.242, -0.199]) |
| 8 | cosine | 12 | 0.237 [0.222, 0.253] | 0.307 [0.282, 0.333] | -0.070 [-0.099, -0.041] (boot [-0.090, -0.048]) |
| 16 | basic | 6 | 0.071 [0.055, 0.087] | 0.333 [0.262, 0.405] | -0.262 [-0.365, -0.160] (boot [-0.277, -0.248]) |
| 16 | cosine | 6 | 0.179 [0.135, 0.223] | 0.333 [0.262, 0.405] | -0.155 [-0.252, -0.058] (boot [-0.174, -0.136]) |

**Any image passes (SSIM ≥ 0.9)**

| K | Attacker | Releases | Attack | Baseline | Attack − baseline |
|---|---|---|---|---|---|
| 1 | basic | 96 | 0.010 [-0.011, 0.032] | 0.000 [0.000, 0.000] | +0.010 [-0.011, 0.032] (boot [0.000, 0.031]) |
| 1 | cosine | 96 | 0.146 [0.074, 0.218] | 0.000 [0.000, 0.000] | **+0.146 [0.074, 0.218] (boot [0.083, 0.219])** |
| 2 | basic | 48 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] (boot [0.000, 0.000]) |
| 2 | cosine | 48 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] (boot [0.000, 0.000]) |
| 4 | basic | 24 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] (boot [0.000, 0.000]) |
| 4 | cosine | 24 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] (boot [0.000, 0.000]) |
| 8 | basic | 12 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] (boot [0.000, 0.000]) |
| 8 | cosine | 12 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] (boot [0.000, 0.000]) |
| 16 | basic | 6 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] (boot [0.000, 0.000]) |
| 16 | cosine | 6 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] (boot [0.000, 0.000]) |

**Retrieval hit (top-1 in class)**

| K | Attacker | Releases | Attack | Baseline | Attack − baseline |
|---|---|---|---|---|---|
| 1 | basic | 96 | 0.281 [0.194, 0.368] | 0.000 [0.000, 0.000] | **+0.281 [0.194, 0.368] (boot [0.198, 0.365])** |
| 1 | cosine | 96 | 0.938 [0.888, 0.987] | 0.000 [0.000, 0.000] | **+0.938 [0.888, 0.987] (boot [0.885, 0.979])** |
| 2 | basic | 48 | 0.479 [0.348, 0.611] | 0.000 [0.000, 0.000] | **+0.479 [0.348, 0.611] (boot [0.354, 0.604])** |
| 2 | cosine | 48 | 0.917 [0.834, 0.999] | 0.000 [0.000, 0.000] | **+0.917 [0.834, 0.999] (boot [0.833, 0.979])** |
| 4 | basic | 24 | 0.583 [0.371, 0.796] | 0.000 [0.000, 0.000] | **+0.583 [0.371, 0.796] (boot [0.417, 0.750])** |
| 4 | cosine | 24 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |
| 8 | basic | 12 | 0.417 [0.104, 0.730] | 0.000 [0.000, 0.000] | **+0.417 [0.104, 0.730] (boot [0.167, 0.583])** |
| 8 | cosine | 12 | 0.917 [0.651, 1.182] | 0.000 [0.000, 0.000] | **+0.917 [0.651, 1.182] (boot [0.750, 1.000])** |
| 16 | basic | 6 | 0.500 [-1.618, 2.618] | 0.000 [0.000, 0.000] | +0.500 [-1.618, 2.618] (boot [0.333, 0.667]) |
| 16 | cosine | 6 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | **+1.000 [1.000, 1.000] (boot [1.000, 1.000])** |

**Share of single reconstructions whose top-1 pick is exactly their own target** (per image, attack only; shows that low-SSIM reconstructions can still identify the image):

| Attacker | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|
| basic | 28% | 30% | 28% | 8% | 4% |
| cosine | 94% | 74% | 58% | 39% | 24% |

Retrieval by chance (one uniform random pick per target in its class): K = 1: 0.10%, K = 2: 0.22%, K = 4: 0.50%, K = 8: 1.26%, K = 16: 3.71%.
Check: recomputed SSIM of the saved reconstructions differs from the saved records by at most 3.9e-05 (float16 storage).

## Gallery sizes (all test images of each class)

| Dataset | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| mnist | 980 | 1135 | 1032 | 1010 | 982 | 892 | 958 | 1028 | 974 | 1009 |
| cifar10 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |

## Limits of this exploratory pass

- **Old images and grouping.** Targets are test 50–81, the same images as the conference study; tuning used 0–49. Not the locked held-out design.
- **Reconstruction labels.** The attack scripts save reconstructions already matched to targets, but not the label each reconstruction was optimized for. Here each reconstruction takes the label of its matched target. This is exact for K = 1 and for groups with one class. For mixed groups, a failed reconstruction may be searched in its matched target's class instead of its own, which can only help the attacker slightly. New runs should save the attacker's label per reconstruction.
- **Same-class guess retrieval is always 0.** That guess is itself a test image (index ≥ 82) and so is in the gallery; its top-1 is itself, never a release image. The class-average guess picks one fixed typical image per class. So the retrieval baseline here is almost always 0; the chance line above is the fairer reference. The locked design must choose guesses that are not gallery members, or exclude a guess's own image.
- **Retrieval control.** Across the MNIST settings at K = 8 and 16, no top-1 pick was a study image (test 50–81) outside its own release, and only 1 of 200 random-noise queries landed in 50–81 (chance about 0.6). So hits are not a bias towards the study images (checked by hand on 21 Nov 2026; not part of this script).
- **Few releases at large K.** K = 8 has 12 releases and K = 16 has 6, over 3 checkpoints.
- **Seeds are checkpoints.** For the fresh settings, a seed is a fresh model, so the strata are fresh initialisations.

Run time: 140 s.
