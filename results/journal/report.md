# Journal results

*Made by `scripts/analyze_journal.py`. Main objective of attacker 1: **cosine**. Attacker 2 uses cosine.*

## 1. Main endpoint

MNIST fresh, N = 2, E = 5, attacker 2 (cosine). Positive control: attacker 1, main objective, E = 1.

**Outcome: more**

| Part | Estimate [interval] | Rule | Result |
|---|---|---|---|
| Positive control (s − b, E = 1) | 0.527 [0.458, 0.597] (95%) | lower end > 0 | passes |
| 1 × 2 beats baseline (s − b) | 0.078 [-0.006, 0.163] (95%) | lower end > 0 | no |
| 2 × 1 beats baseline (s − b) | 0.524 [0.453, 0.594] (95%) | lower end > 0 | yes |
| Gap 2 × 1 − 1 × 2 | 0.445 [0.357, 0.534] (90%) | margin ±0.05 | n = 30 |

Bootstrap check (report only): gap 90% 0.445 [0.362, 0.523]; control 95% 0.527 [0.462, 0.588].

Sentence: "Under attacker 2, on MNIST fresh at N = 2 and E = 5, the 2-client sum gave better reconstructions than the 1-client update (gap 0.445, 90% interval [0.357, 0.534], 30 paired releases on 3 checkpoints)."

## 2. Secondary labels (E = 5)

Same rules, each (setting, N) with its own positive control (attacker 1, main objective, E = 1). Not adjusted for multiple tests.

| Setting | N | Attacker | Outcome | PC s − b (95%) | 1×N s − b (95%) | N×1 s − b (95%) | Gap (90%) |
|---|---|---|---|---|---|---|---|
| mnist-fresh | 2 | attacker 2, cosine | more | 0.527 [0.458, 0.597] | 0.078 [-0.006, 0.163] | 0.524 [0.453, 0.594] | 0.445 [0.357, 0.534] |
| mnist-fresh | 2 | attacker 1, cosine | more | 0.527 [0.458, 0.597] | -0.150 [-0.253, -0.047] | 0.525 [0.454, 0.595] | 0.674 [0.580, 0.769] |
| mnist-fresh | 2 | attacker 1, basic | neither above baseline | 0.527 [0.458, 0.597] | -0.295 [-0.328, -0.261] | -0.283 [-0.314, -0.251] | 0.012 [0.003, 0.021] |
| mnist-trained | 2 | attacker 2, cosine | more | 0.298 [0.250, 0.346] | 0.023 [-0.053, 0.099] | 0.366 [0.330, 0.402] | 0.343 [0.283, 0.402] |
| mnist-trained | 2 | attacker 1, cosine | inconclusive | 0.298 [0.250, 0.346] | 0.200 [0.107, 0.293] | 0.317 [0.281, 0.353] | 0.117 [0.047, 0.188] |
| mnist-trained | 2 | attacker 1, basic | neither above baseline | 0.298 [0.250, 0.346] | -0.093 [-0.206, 0.019] | -0.143 [-0.227, -0.059] | -0.050 [-0.134, 0.034] |
| mnist-fresh | 4 | attacker 2, cosine | more | 0.467 [0.406, 0.529] | -0.165 [-0.212, -0.118] | 0.465 [0.403, 0.527] | 0.630 [0.567, 0.692] |
| mnist-fresh | 4 | attacker 1, cosine | more | 0.467 [0.406, 0.529] | -0.269 [-0.293, -0.246] | 0.462 [0.398, 0.525] | 0.731 [0.685, 0.778] |
| mnist-fresh | 4 | attacker 1, basic | neither above baseline | 0.467 [0.406, 0.529] | -0.279 [-0.303, -0.255] | -0.264 [-0.291, -0.237] | 0.015 [0.007, 0.024] |
| mnist-trained | 4 | attacker 2, cosine | more | 0.126 [0.088, 0.164] | -0.031 [-0.087, 0.024] | 0.179 [0.137, 0.221] | 0.210 [0.154, 0.267] |
| mnist-trained | 4 | attacker 1, cosine | more | 0.126 [0.088, 0.164] | 0.038 [-0.029, 0.106] | 0.166 [0.122, 0.210] | 0.128 [0.080, 0.176] |
| mnist-trained | 4 | attacker 1, basic | neither above baseline | 0.126 [0.088, 0.164] | -0.118 [-0.197, -0.040] | -0.102 [-0.173, -0.032] | 0.016 [-0.038, 0.070] |

## 3. Second endpoint: any image passes (SSIM ≥ 0.9)

| Setting | N | E | Split | Attacker | Rate | Rate − baseline (95%) |
|---|---|---|---|---|---|---|
| mnist-fresh | 2 | 1 | 1x2 | attacker 1, cosine | 0.867 | 0.860 [0.729, 0.991] |
| mnist-fresh | 2 | 1 | 1x2 | attacker 1, basic | 1.000 | 0.993 [0.978, 1.008] |
| mnist-fresh | 2 | 5 | 1x2 | attacker 1, cosine | 0.100 | 0.093 [-0.026, 0.213] |
| mnist-fresh | 2 | 5 | 1x2 | attacker 1, basic | 0.000 | -0.007 [-0.022, 0.008] |
| mnist-fresh | 2 | 5 | 1x2 | attacker 2, cosine | 0.100 | 0.093 [-0.026, 0.213] |
| mnist-fresh | 2 | 5 | 2x1 | attacker 1, cosine | 0.867 | 0.860 [0.729, 0.991] |
| mnist-fresh | 2 | 5 | 2x1 | attacker 1, basic | 0.000 | -0.007 [-0.022, 0.008] |
| mnist-fresh | 2 | 5 | 2x1 | attacker 2, cosine | 0.900 | 0.893 [0.777, 1.010] |
| mnist-fresh | 4 | 1 | 1x4 | attacker 1, cosine | 0.767 | 0.753 [0.592, 0.914] |
| mnist-fresh | 4 | 1 | 1x4 | attacker 1, basic | 1.000 | 0.987 [0.967, 1.006] |
| mnist-fresh | 4 | 5 | 1x4 | attacker 1, cosine | 0.000 | -0.013 [-0.033, 0.006] |
| mnist-fresh | 4 | 5 | 1x4 | attacker 1, basic | 0.000 | -0.013 [-0.033, 0.006] |
| mnist-fresh | 4 | 5 | 1x4 | attacker 2, cosine | 0.000 | -0.013 [-0.033, 0.006] |
| mnist-fresh | 4 | 5 | 4x1 | attacker 1, cosine | 0.800 | 0.787 [0.634, 0.940] |
| mnist-fresh | 4 | 5 | 4x1 | attacker 1, basic | 0.000 | -0.013 [-0.033, 0.006] |
| mnist-fresh | 4 | 5 | 4x1 | attacker 2, cosine | 0.800 | 0.787 [0.634, 0.940] |
| mnist-trained | 2 | 1 | 1x2 | attacker 1, cosine | 0.500 | 0.493 [0.310, 0.677] |
| mnist-trained | 2 | 1 | 1x2 | attacker 1, basic | 0.433 | 0.427 [0.235, 0.618] |
| mnist-trained | 2 | 5 | 1x2 | attacker 1, cosine | 0.300 | 0.293 [0.118, 0.469] |
| mnist-trained | 2 | 5 | 1x2 | attacker 1, basic | 0.033 | 0.027 [-0.049, 0.103] |
| mnist-trained | 2 | 5 | 1x2 | attacker 2, cosine | 0.100 | 0.093 [-0.027, 0.214] |
| mnist-trained | 2 | 5 | 2x1 | attacker 1, cosine | 0.600 | 0.593 [0.405, 0.782] |
| mnist-trained | 2 | 5 | 2x1 | attacker 1, basic | 0.033 | 0.027 [-0.049, 0.103] |
| mnist-trained | 2 | 5 | 2x1 | attacker 2, cosine | 0.633 | 0.627 [0.439, 0.814] |
| mnist-trained | 4 | 1 | 1x4 | attacker 1, cosine | 0.267 | 0.253 [0.091, 0.415] |
| mnist-trained | 4 | 1 | 1x4 | attacker 1, basic | 0.433 | 0.420 [0.230, 0.610] |
| mnist-trained | 4 | 5 | 1x4 | attacker 1, cosine | 0.100 | 0.087 [-0.035, 0.208] |
| mnist-trained | 4 | 5 | 1x4 | attacker 1, basic | 0.000 | -0.013 [-0.033, 0.006] |
| mnist-trained | 4 | 5 | 1x4 | attacker 2, cosine | 0.033 | 0.020 [-0.057, 0.097] |
| mnist-trained | 4 | 5 | 4x1 | attacker 1, cosine | 0.367 | 0.353 [0.164, 0.543] |
| mnist-trained | 4 | 5 | 4x1 | attacker 1, basic | 0.033 | 0.020 [-0.059, 0.099] |
| mnist-trained | 4 | 5 | 4x1 | attacker 2, cosine | 0.300 | 0.287 [0.109, 0.464] |

| Setting | N | Attacker | Gap N×1 − 1×N (90%) | (95%) | Both / only N×1 / only 1×N / neither |
|---|---|---|---|---|---|
| mnist-fresh | 2 | attacker 1, cosine | 0.767 [0.643, 0.890] | 0.767 [0.617, 0.917] | 3 / 23 / 0 / 4 |
| mnist-fresh | 2 | attacker 1, basic | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0 / 0 / 0 / 30 |
| mnist-fresh | 2 | attacker 2, cosine | 0.800 [0.686, 0.914] | 0.800 [0.662, 0.938] | 3 / 24 / 0 / 3 |
| mnist-fresh | 4 | attacker 1, cosine | 0.800 [0.671, 0.929] | 0.800 [0.645, 0.955] | 0 / 24 / 0 / 6 |
| mnist-fresh | 4 | attacker 1, basic | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0 / 0 / 0 / 30 |
| mnist-fresh | 4 | attacker 2, cosine | 0.800 [0.671, 0.929] | 0.800 [0.645, 0.955] | 0 / 24 / 0 / 6 |
| mnist-trained | 2 | attacker 1, cosine | 0.300 [0.136, 0.464] | 0.300 [0.102, 0.498] | 8 / 10 / 1 / 11 |
| mnist-trained | 2 | attacker 1, basic | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1 / 0 / 0 / 29 |
| mnist-trained | 2 | attacker 2, cosine | 0.533 [0.375, 0.692] | 0.533 [0.343, 0.724] | 3 / 16 / 0 / 11 |
| mnist-trained | 4 | attacker 1, cosine | 0.267 [0.098, 0.435] | 0.267 [0.064, 0.470] | 2 / 9 / 1 / 18 |
| mnist-trained | 4 | attacker 1, basic | 0.033 [-0.028, 0.094] | 0.033 [-0.042, 0.109] | 0 / 1 / 0 / 29 |
| mnist-trained | 4 | attacker 2, cosine | 0.267 [0.104, 0.430] | 0.267 [0.070, 0.463] | 0 / 9 / 1 / 20 |

## 4. Averages hide the risk

| Setting | N | E | Split | Attacker | Releases | Mean SSIM | Any passes | Retrieval hit | SSIM per checkpoint |
|---|---|---|---|---|---|---|---|---|---|
| mnist-fresh | 2 | 1 | 1x2 | attacker 1, cosine | 30 | 0.859 | 0.867 | 1.000 | k0: 0.913, k1: 0.774, k2: 0.890 |
| mnist-fresh | 2 | 1 | 1x2 | attacker 1, basic | 30 | 0.950 | 1.000 | 1.000 | k0: 0.992, k1: 0.865, k2: 0.993 |
| mnist-fresh | 2 | 5 | 1x2 | attacker 1, cosine | 30 | 0.182 | 0.100 | 0.333 | k0: 0.152, k1: 0.056, k2: 0.336 |
| mnist-fresh | 2 | 5 | 1x2 | attacker 1, basic | 30 | 0.037 | 0.000 | 0.067 | k0: 0.065, k1: 0.020, k2: 0.025 |
| mnist-fresh | 2 | 5 | 1x2 | attacker 2, cosine | 30 | 0.410 | 0.100 | 0.933 | k0: 0.271, k1: 0.309, k2: 0.650 |
| mnist-fresh | 2 | 5 | 2x1 | attacker 1, cosine | 30 | 0.856 | 0.867 | 1.000 | k0: 0.906, k1: 0.770, k2: 0.892 |
| mnist-fresh | 2 | 5 | 2x1 | attacker 1, basic | 30 | 0.049 | 0.000 | 0.067 | k0: 0.084, k1: 0.028, k2: 0.034 |
| mnist-fresh | 2 | 5 | 2x1 | attacker 2, cosine | 30 | 0.855 | 0.900 | 1.000 | k0: 0.908, k1: 0.772, k2: 0.885 |
| mnist-fresh | 4 | 1 | 1x4 | attacker 1, cosine | 30 | 0.780 | 0.767 | 1.000 | k0: 0.725, k1: 0.806, k2: 0.809 |
| mnist-fresh | 4 | 1 | 1x4 | attacker 1, basic | 30 | 0.942 | 1.000 | 1.000 | k0: 0.949, k1: 0.909, k2: 0.969 |
| mnist-fresh | 4 | 5 | 1x4 | attacker 1, cosine | 30 | 0.043 | 0.000 | 0.067 | k0: 0.041, k1: 0.032, k2: 0.057 |
| mnist-fresh | 4 | 5 | 1x4 | attacker 1, basic | 30 | 0.034 | 0.000 | 0.000 | k0: 0.040, k1: 0.028, k2: 0.033 |
| mnist-fresh | 4 | 5 | 1x4 | attacker 2, cosine | 30 | 0.148 | 0.000 | 0.467 | k0: 0.106, k1: 0.122, k2: 0.216 |
| mnist-fresh | 4 | 5 | 4x1 | attacker 1, cosine | 30 | 0.775 | 0.800 | 1.000 | k0: 0.725, k1: 0.801, k2: 0.799 |
| mnist-fresh | 4 | 5 | 4x1 | attacker 1, basic | 30 | 0.049 | 0.000 | 0.033 | k0: 0.059, k1: 0.033, k2: 0.055 |
| mnist-fresh | 4 | 5 | 4x1 | attacker 2, cosine | 30 | 0.778 | 0.800 | 1.000 | k0: 0.725, k1: 0.802, k2: 0.806 |
| mnist-trained | 2 | 1 | 1x2 | attacker 1, cosine | 30 | 0.630 | 0.500 | 0.967 | k0: 0.614, k1: 0.634, k2: 0.641 |
| mnist-trained | 2 | 1 | 1x2 | attacker 1, basic | 30 | 0.397 | 0.433 | 0.600 | k0: 0.318, k1: 0.473, k2: 0.400 |
| mnist-trained | 2 | 5 | 1x2 | attacker 1, cosine | 30 | 0.531 | 0.300 | 0.833 | k0: 0.499, k1: 0.464, k2: 0.630 |
| mnist-trained | 2 | 5 | 1x2 | attacker 1, basic | 30 | 0.238 | 0.033 | 0.400 | k0: 0.284, k1: 0.212, k2: 0.217 |
| mnist-trained | 2 | 5 | 1x2 | attacker 2, cosine | 30 | 0.355 | 0.100 | 0.733 | k0: 0.309, k1: 0.350, k2: 0.405 |
| mnist-trained | 2 | 5 | 2x1 | attacker 1, cosine | 30 | 0.648 | 0.600 | 1.000 | k0: 0.609, k1: 0.669, k2: 0.666 |
| mnist-trained | 2 | 5 | 2x1 | attacker 1, basic | 30 | 0.188 | 0.033 | 0.367 | k0: 0.166, k1: 0.278, k2: 0.121 |
| mnist-trained | 2 | 5 | 2x1 | attacker 2, cosine | 30 | 0.697 | 0.633 | 1.000 | k0: 0.643, k1: 0.698, k2: 0.751 |
| mnist-trained | 4 | 1 | 1x4 | attacker 1, cosine | 30 | 0.439 | 0.267 | 1.000 | k0: 0.404, k1: 0.476, k2: 0.437 |
| mnist-trained | 4 | 1 | 1x4 | attacker 1, basic | 30 | 0.360 | 0.433 | 0.667 | k0: 0.244, k1: 0.395, k2: 0.442 |
| mnist-trained | 4 | 5 | 1x4 | attacker 1, cosine | 30 | 0.351 | 0.100 | 0.700 | k0: 0.294, k1: 0.488, k2: 0.271 |
| mnist-trained | 4 | 5 | 1x4 | attacker 1, basic | 30 | 0.194 | 0.000 | 0.400 | k0: 0.218, k1: 0.270, k2: 0.095 |
| mnist-trained | 4 | 5 | 1x4 | attacker 2, cosine | 30 | 0.281 | 0.033 | 0.833 | k0: 0.309, k1: 0.272, k2: 0.263 |
| mnist-trained | 4 | 5 | 4x1 | attacker 1, cosine | 30 | 0.478 | 0.367 | 1.000 | k0: 0.450, k1: 0.535, k2: 0.450 |
| mnist-trained | 4 | 5 | 4x1 | attacker 1, basic | 30 | 0.210 | 0.033 | 0.533 | k0: 0.146, k1: 0.242, k2: 0.243 |
| mnist-trained | 4 | 5 | 4x1 | attacker 2, cosine | 30 | 0.492 | 0.300 | 1.000 | k0: 0.477, k1: 0.532, k2: 0.467 |
| label-aware baseline | 2 | – | – | max(class average, same class) | 30 | 0.331 | 0.007 | 0.000 | – |
| label-aware baseline | 4 | – | – | max(class average, same class) | 30 | 0.313 | 0.013 | 0.040 | – |

Baseline parts and context (mean SSIM): N = 2: class average 0.326, same class 0.273, exact retrieval chance 0.0022, all black 0.112, average digit 0.120, random noise 0.014; N = 4: class average 0.310, same class 0.279, exact retrieval chance 0.0057, all black 0.090, average digit 0.118, random noise 0.027

## 5. Drift (E = 5)

| Setting | N | Releases | Median | Range | Median per checkpoint | Median norm ratio | E = 1 drift, min |
|---|---|---|---|---|---|---|---|
| mnist-fresh | 2 | 30 | 0.8530 | 0.3910–1.0000 | k0: 0.853, k1: 0.848, k2: 0.847 | 0.8607 | 1.000000 |
| mnist-trained | 2 | 30 | 0.9950 | -0.2764–1.0000 | k0: 0.997, k1: 0.959, k2: 0.998 | 0.7797 | 1.000000 |
| mnist-fresh | 4 | 30 | 0.7353 | -0.1429–0.8792 | k0: 0.707, k1: 0.759, k2: 0.743 | 0.7824 | 1.000000 |
| mnist-trained | 4 | 30 | 0.9445 | -0.1549–0.9997 | k0: 0.643, k1: 0.970, k2: 0.103 | 0.6107 | 1.000000 |

## 6. Secondary objective (attacker 1, basic)

| Setting | N | E | Split | Mean SSIM | s − b (95%) | Any passes |
|---|---|---|---|---|---|---|
| mnist-fresh | 2 | 1 | 1x2 | 0.950 | 0.619 [0.575, 0.662] | 1.000 |
| mnist-fresh | 2 | 5 | 1x2 | 0.037 | -0.295 [-0.328, -0.261] | 0.000 |
| mnist-fresh | 2 | 5 | 2x1 | 0.049 | -0.283 [-0.314, -0.251] | 0.000 |
| mnist-fresh | 4 | 1 | 1x4 | 0.942 | 0.630 [0.589, 0.670] | 1.000 |
| mnist-fresh | 4 | 5 | 1x4 | 0.034 | -0.279 [-0.303, -0.255] | 0.000 |
| mnist-fresh | 4 | 5 | 4x1 | 0.049 | -0.264 [-0.291, -0.237] | 0.000 |
| mnist-trained | 2 | 1 | 1x2 | 0.397 | 0.066 [-0.075, 0.208] | 0.433 |
| mnist-trained | 2 | 5 | 1x2 | 0.238 | -0.093 [-0.206, 0.019] | 0.033 |
| mnist-trained | 2 | 5 | 2x1 | 0.188 | -0.143 [-0.227, -0.059] | 0.033 |
| mnist-trained | 4 | 1 | 1x4 | 0.360 | 0.048 [-0.056, 0.151] | 0.433 |
| mnist-trained | 4 | 5 | 1x4 | 0.194 | -0.118 [-0.197, -0.040] | 0.000 |
| mnist-trained | 4 | 5 | 4x1 | 0.210 | -0.102 [-0.173, -0.032] | 0.033 |

## 7. Retrieval (top-1 in the slot label's class gallery)

Class gallery sizes (every test image of the class): class 0: 980, class 1: 1,135, class 2: 1,032, class 3: 1,010, class 4: 982, class 5: 892, class 6: 958, class 7: 1,028, class 8: 974, class 9: 1,009.

| Setting | N | E | Split | Attacker | Hit rate | Hit − baseline (95%) | Per checkpoint |
|---|---|---|---|---|---|---|---|
| mnist-fresh | 2 | 1 | 1x2 | attacker 1, cosine | 1.000 | 1.000 [1.000, 1.000] | k0: 1.000, k1: 1.000, k2: 1.000 |
| mnist-fresh | 2 | 1 | 1x2 | attacker 1, basic | 1.000 | 1.000 [1.000, 1.000] | k0: 1.000, k1: 1.000, k2: 1.000 |
| mnist-fresh | 2 | 5 | 1x2 | attacker 1, cosine | 0.333 | 0.333 [0.159, 0.508] | k0: 0.500, k1: 0.100, k2: 0.400 |
| mnist-fresh | 2 | 5 | 1x2 | attacker 1, basic | 0.067 | 0.067 [-0.034, 0.167] | k0: 0.200, k1: 0.000, k2: 0.000 |
| mnist-fresh | 2 | 5 | 1x2 | attacker 2, cosine | 0.933 | 0.933 [0.833, 1.034] | k0: 0.800, k1: 1.000, k2: 1.000 |
| mnist-fresh | 2 | 5 | 2x1 | attacker 1, cosine | 1.000 | 1.000 [1.000, 1.000] | k0: 1.000, k1: 1.000, k2: 1.000 |
| mnist-fresh | 2 | 5 | 2x1 | attacker 1, basic | 0.067 | 0.067 [-0.034, 0.167] | k0: 0.200, k1: 0.000, k2: 0.000 |
| mnist-fresh | 2 | 5 | 2x1 | attacker 2, cosine | 1.000 | 1.000 [1.000, 1.000] | k0: 1.000, k1: 1.000, k2: 1.000 |
| mnist-fresh | 4 | 1 | 1x4 | attacker 1, cosine | 1.000 | 0.960 [0.884, 1.036] | k0: 1.000, k1: 1.000, k2: 1.000 |
| mnist-fresh | 4 | 1 | 1x4 | attacker 1, basic | 1.000 | 0.960 [0.884, 1.036] | k0: 1.000, k1: 1.000, k2: 1.000 |
| mnist-fresh | 4 | 5 | 1x4 | attacker 1, cosine | 0.067 | 0.027 [-0.049, 0.103] | k0: 0.000, k1: 0.000, k2: 0.200 |
| mnist-fresh | 4 | 5 | 1x4 | attacker 1, basic | 0.000 | -0.040 [-0.116, 0.036] | k0: 0.000, k1: 0.000, k2: 0.000 |
| mnist-fresh | 4 | 5 | 1x4 | attacker 2, cosine | 0.467 | 0.427 [0.237, 0.617] | k0: 0.500, k1: 0.500, k2: 0.400 |
| mnist-fresh | 4 | 5 | 4x1 | attacker 1, cosine | 1.000 | 0.960 [0.884, 1.036] | k0: 1.000, k1: 1.000, k2: 1.000 |
| mnist-fresh | 4 | 5 | 4x1 | attacker 1, basic | 0.033 | -0.007 [-0.106, 0.093] | k0: 0.100, k1: 0.000, k2: 0.000 |
| mnist-fresh | 4 | 5 | 4x1 | attacker 2, cosine | 1.000 | 0.960 [0.884, 1.036] | k0: 1.000, k1: 1.000, k2: 1.000 |
| mnist-trained | 2 | 1 | 1x2 | attacker 1, cosine | 0.967 | 0.967 [0.891, 1.042] | k0: 1.000, k1: 1.000, k2: 0.900 |
| mnist-trained | 2 | 1 | 1x2 | attacker 1, basic | 0.600 | 0.600 [0.409, 0.791] | k0: 0.600, k1: 0.700, k2: 0.500 |
| mnist-trained | 2 | 5 | 1x2 | attacker 1, cosine | 0.833 | 0.833 [0.687, 0.980] | k0: 0.800, k1: 0.900, k2: 0.800 |
| mnist-trained | 2 | 5 | 1x2 | attacker 1, basic | 0.400 | 0.400 [0.209, 0.591] | k0: 0.500, k1: 0.300, k2: 0.400 |
| mnist-trained | 2 | 5 | 1x2 | attacker 2, cosine | 0.733 | 0.733 [0.562, 0.904] | k0: 0.800, k1: 0.800, k2: 0.600 |
| mnist-trained | 2 | 5 | 2x1 | attacker 1, cosine | 1.000 | 1.000 [1.000, 1.000] | k0: 1.000, k1: 1.000, k2: 1.000 |
| mnist-trained | 2 | 5 | 2x1 | attacker 1, basic | 0.367 | 0.367 [0.180, 0.553] | k0: 0.300, k1: 0.500, k2: 0.300 |
| mnist-trained | 2 | 5 | 2x1 | attacker 2, cosine | 1.000 | 1.000 [1.000, 1.000] | k0: 1.000, k1: 1.000, k2: 1.000 |
| mnist-trained | 4 | 1 | 1x4 | attacker 1, cosine | 1.000 | 0.960 [0.884, 1.036] | k0: 1.000, k1: 1.000, k2: 1.000 |
| mnist-trained | 4 | 1 | 1x4 | attacker 1, basic | 0.667 | 0.627 [0.439, 0.814] | k0: 0.600, k1: 0.600, k2: 0.800 |
| mnist-trained | 4 | 5 | 1x4 | attacker 1, cosine | 0.700 | 0.660 [0.461, 0.859] | k0: 0.700, k1: 0.900, k2: 0.500 |
| mnist-trained | 4 | 5 | 1x4 | attacker 1, basic | 0.400 | 0.360 [0.159, 0.561] | k0: 0.500, k1: 0.500, k2: 0.200 |
| mnist-trained | 4 | 5 | 1x4 | attacker 2, cosine | 0.833 | 0.793 [0.646, 0.941] | k0: 1.000, k1: 0.800, k2: 0.700 |
| mnist-trained | 4 | 5 | 4x1 | attacker 1, cosine | 1.000 | 0.960 [0.884, 1.036] | k0: 1.000, k1: 1.000, k2: 1.000 |
| mnist-trained | 4 | 5 | 4x1 | attacker 1, basic | 0.533 | 0.493 [0.300, 0.686] | k0: 0.400, k1: 0.600, k2: 0.600 |
| mnist-trained | 4 | 5 | 4x1 | attacker 2, cosine | 1.000 | 0.960 [0.884, 1.036] | k0: 1.000, k1: 1.000, k2: 1.000 |

| Setting | N | Attacker | Hit gap N×1 − 1×N (90%) | (95%) | Both / only N×1 / only 1×N / neither |
|---|---|---|---|---|---|
| mnist-fresh | 2 | attacker 1, cosine | 0.667 [0.522, 0.812] | 0.667 [0.492, 0.841] | 10 / 20 / 0 / 0 |
| mnist-fresh | 2 | attacker 1, basic | 0.000 [-0.091, 0.091] | 0.000 [-0.112, 0.112] | 1 / 1 / 1 / 27 |
| mnist-fresh | 2 | attacker 2, cosine | 0.067 [-0.015, 0.148] | 0.067 [-0.034, 0.167] | 28 / 2 / 0 / 0 |
| mnist-fresh | 4 | attacker 1, cosine | 0.933 [0.852, 1.015] | 0.933 [0.833, 1.034] | 2 / 28 / 0 / 0 |
| mnist-fresh | 4 | attacker 1, basic | 0.033 [-0.028, 0.094] | 0.033 [-0.042, 0.109] | 0 / 1 / 0 / 29 |
| mnist-fresh | 4 | attacker 2, cosine | 0.533 [0.371, 0.696] | 0.533 [0.337, 0.729] | 14 / 16 / 0 / 0 |
| mnist-trained | 2 | attacker 1, cosine | 0.167 [0.045, 0.288] | 0.167 [0.020, 0.313] | 25 / 5 / 0 / 0 |
| mnist-trained | 2 | attacker 1, basic | -0.033 [-0.204, 0.138] | -0.033 [-0.239, 0.173] | 7 / 4 / 5 / 14 |
| mnist-trained | 2 | attacker 2, cosine | 0.267 [0.125, 0.408] | 0.267 [0.096, 0.438] | 22 / 8 / 0 / 0 |
| mnist-trained | 4 | attacker 1, cosine | 0.300 [0.159, 0.441] | 0.300 [0.130, 0.470] | 21 / 9 / 0 / 0 |
| mnist-trained | 4 | attacker 1, basic | 0.133 [-0.017, 0.283] | 0.133 [-0.048, 0.314] | 10 / 6 / 2 / 12 |
| mnist-trained | 4 | attacker 2, cosine | 0.167 [0.049, 0.284] | 0.167 [0.024, 0.309] | 25 / 5 / 0 / 0 |

## 8. Drift and labels (pre-declared secondary)

On the fresh model a same-label pair barely drifts, so it acts like E = 1. Secondary: this never changes the main outcome.

Main gap (MNIST fresh, N = 2, E = 5, attacker 2), 2 × 1 − 1 × 2, split by the release's labels:

| Releases | Paired n | Per checkpoint | Median drift | Gap (90%) | Gap (95%) |
|---|---|---|---|---|---|
| same label | 4 | k0: 1, k1: 0, k2: 3 | 1.0000 | descriptive only: -0.012 [-0.013, -0.012] bootstrap; releases k0: 1, k1: 0, k2: 3 | descriptive only: -0.012 [-0.013, -0.012] bootstrap; releases k0: 1, k1: 0, k2: 3 |
| different labels | 26 | k0: 9, k1: 10, k2: 7 | 0.8382 | 0.516 [0.432, 0.600] | 0.516 [0.414, 0.617] |

Drift per release, mnist-fresh, N = 2 (E = 5):

| Release | Checkpoint | Labels | Same label | Drift | Norm ratio |
|---|---|---|---|---|---|
| 0 | 0 | 6, 3 | no | 0.8478 | 0.8491 |
| 1 | 0 | 7, 7 | yes | 1.0000 | 1.0000 |
| 2 | 0 | 3, 4 | no | 0.9350 | 0.9306 |
| 3 | 0 | 6, 1 | no | 0.9419 | 0.9300 |
| 4 | 0 | 3, 6 | no | 0.8359 | 0.8362 |
| 5 | 0 | 9, 3 | no | 0.8858 | 0.8877 |
| 6 | 0 | 1, 4 | no | 0.8399 | 0.8083 |
| 7 | 0 | 7, 4 | no | 0.4594 | 0.7052 |
| 8 | 0 | 4, 9 | no | 0.8581 | 0.8478 |
| 9 | 0 | 3, 5 | no | 0.8364 | 0.8727 |
| 10 | 1 | 8, 9 | no | 0.3910 | 0.7576 |
| 11 | 1 | 9, 2 | no | 0.3961 | 0.8848 |
| 12 | 1 | 0, 9 | no | 0.4996 | 0.7651 |
| 13 | 1 | 0, 7 | no | 0.8613 | 0.8593 |
| 14 | 1 | 0, 5 | no | 0.8344 | 0.8419 |
| 15 | 1 | 9, 7 | no | 0.5063 | 0.8044 |
| 16 | 1 | 2, 7 | no | 0.9238 | 0.9132 |
| 17 | 1 | 6, 0 | no | 0.8713 | 0.8728 |
| 18 | 1 | 3, 7 | no | 0.9005 | 0.9055 |
| 19 | 1 | 7, 2 | no | 0.9080 | 0.8925 |
| 20 | 2 | 1, 1 | yes | 1.0000 | 1.0000 |
| 21 | 2 | 1, 7 | no | 0.4612 | 0.6953 |
| 22 | 2 | 1, 1 | yes | 1.0000 | 1.0000 |
| 23 | 2 | 7, 3 | no | 0.4964 | 0.7441 |
| 24 | 2 | 9, 3 | no | 0.8294 | 0.8404 |
| 25 | 2 | 9, 9 | yes | 1.0000 | 1.0000 |
| 26 | 2 | 7, 6 | no | 0.4870 | 0.6693 |
| 27 | 2 | 4, 5 | no | 0.7934 | 0.7996 |
| 28 | 2 | 1, 6 | no | 0.8971 | 0.8800 |
| 29 | 2 | 5, 0 | no | 0.8650 | 0.8621 |

Drift per release, mnist-trained, N = 2 (E = 5):

| Release | Checkpoint | Labels | Same label | Drift | Norm ratio |
|---|---|---|---|---|---|
| 0 | 0 | 6, 3 | no | 0.9947 | 0.9865 |
| 1 | 0 | 7, 7 | yes | 1.0000 | 1.0001 |
| 2 | 0 | 3, 4 | no | -0.1138 | 0.2877 |
| 3 | 0 | 6, 1 | no | 0.9990 | 0.7704 |
| 4 | 0 | 3, 6 | no | 0.9990 | 0.8171 |
| 5 | 0 | 9, 3 | no | 0.9999 | 0.7111 |
| 6 | 0 | 1, 4 | no | 0.9997 | 0.7382 |
| 7 | 0 | 7, 4 | no | 0.9731 | 1.0909 |
| 8 | 0 | 4, 9 | no | 0.9073 | 0.5051 |
| 9 | 0 | 3, 5 | no | 0.2399 | 0.5392 |
| 10 | 1 | 8, 9 | no | 0.9868 | 1.0154 |
| 11 | 1 | 9, 2 | no | 0.9954 | 0.9742 |
| 12 | 1 | 0, 9 | no | 0.4189 | 0.4765 |
| 13 | 1 | 0, 7 | no | 0.1547 | 0.3145 |
| 14 | 1 | 0, 5 | no | -0.2764 | 0.1307 |
| 15 | 1 | 9, 7 | no | 0.9317 | 0.9729 |
| 16 | 1 | 2, 7 | no | 0.9991 | 1.0163 |
| 17 | 1 | 6, 0 | no | 0.9990 | 0.7408 |
| 18 | 1 | 3, 7 | no | 0.9928 | 0.8974 |
| 19 | 1 | 7, 2 | no | 0.4238 | 0.6573 |
| 20 | 2 | 1, 1 | yes | 0.9999 | 0.9421 |
| 21 | 2 | 1, 7 | no | 0.9971 | 0.7716 |
| 22 | 2 | 1, 1 | yes | 0.9985 | 0.9326 |
| 23 | 2 | 7, 3 | no | 0.9988 | 0.7151 |
| 24 | 2 | 9, 3 | no | 0.6591 | 0.9614 |
| 25 | 2 | 9, 9 | yes | 1.0000 | 1.0001 |
| 26 | 2 | 7, 6 | no | 0.9988 | 0.7793 |
| 27 | 2 | 4, 5 | no | 0.9775 | 0.7801 |
| 28 | 2 | 1, 6 | no | 0.9962 | 1.0542 |
| 29 | 2 | 5, 0 | no | 0.2208 | 0.2702 |

Drift per release, mnist-fresh, N = 4 (E = 5):

| Release | Checkpoint | Labels | Same label | Drift | Norm ratio |
|---|---|---|---|---|---|
| 0 | 0 | 4, 3, 3, 2 | no | 0.7231 | 0.7422 |
| 1 | 0 | 4, 1, 4, 0 | no | 0.8567 | 0.8532 |
| 2 | 0 | 9, 3, 7, 3 | no | 0.3656 | 0.6635 |
| 3 | 0 | 1, 4, 9, 4 | no | 0.8097 | 0.8087 |
| 4 | 0 | 0, 8, 4, 8 | no | 0.6905 | 0.6965 |
| 5 | 0 | 2, 2, 2, 8 | no | 0.8402 | 0.8255 |
| 6 | 0 | 7, 3, 9, 7 | no | 0.2899 | 0.5839 |
| 7 | 0 | 8, 8, 6, 0 | no | 0.7247 | 0.7240 |
| 8 | 0 | 9, 7, 7, 8 | no | 0.0129 | 0.5373 |
| 9 | 0 | 3, 7, 7, 2 | no | -0.1429 | 0.4471 |
| 10 | 1 | 3, 0, 7, 3 | no | 0.7920 | 0.8358 |
| 11 | 1 | 1, 9, 2, 5 | no | 0.3738 | 0.7200 |
| 12 | 1 | 2, 2, 2, 4 | no | 0.8694 | 0.8752 |
| 13 | 1 | 2, 3, 2, 5 | no | 0.7651 | 0.8164 |
| 14 | 1 | 1, 7, 2, 6 | no | 0.7712 | 0.7813 |
| 15 | 1 | 6, 1, 1, 6 | no | 0.8042 | 0.8022 |
| 16 | 1 | 5, 9, 0, 4 | no | 0.2830 | 0.6659 |
| 17 | 1 | 5, 4, 2, 8 | no | 0.7458 | 0.7836 |
| 18 | 1 | 2, 9, 0, 8 | no | 0.4556 | 0.7747 |
| 19 | 1 | 1, 4, 2, 3 | no | 0.7523 | 0.7886 |
| 20 | 2 | 8, 4, 0, 2 | no | 0.7934 | 0.7981 |
| 21 | 2 | 9, 9, 6, 6 | no | 0.8415 | 0.8337 |
| 22 | 2 | 1, 7, 4, 1 | no | 0.5394 | 0.6914 |
| 23 | 2 | 8, 1, 0, 7 | no | 0.4071 | 0.5840 |
| 24 | 2 | 2, 5, 1, 1 | no | 0.8413 | 0.8419 |
| 25 | 2 | 0, 7, 0, 1 | no | 0.6362 | 0.8009 |
| 26 | 2 | 4, 4, 0, 5 | no | 0.7741 | 0.7864 |
| 27 | 2 | 5, 8, 0, 4 | no | 0.7101 | 0.7070 |
| 28 | 2 | 6, 3, 5, 5 | no | 0.8792 | 0.8843 |
| 29 | 2 | 9, 8, 2, 3 | no | 0.7127 | 0.7402 |

Drift per release, mnist-trained, N = 4 (E = 5):

| Release | Checkpoint | Labels | Same label | Drift | Norm ratio |
|---|---|---|---|---|---|
| 0 | 0 | 4, 3, 3, 2 | no | 0.9865 | 0.5597 |
| 1 | 0 | 4, 1, 4, 0 | no | 0.2472 | 0.3394 |
| 2 | 0 | 9, 3, 7, 3 | no | -0.0003 | 0.0997 |
| 3 | 0 | 1, 4, 9, 4 | no | 0.8740 | 0.6642 |
| 4 | 0 | 0, 8, 4, 8 | no | 0.4123 | 0.0983 |
| 5 | 0 | 2, 2, 2, 8 | no | 0.9820 | 0.7077 |
| 6 | 0 | 7, 3, 9, 7 | no | 0.0695 | 0.3285 |
| 7 | 0 | 8, 8, 6, 0 | no | 0.2133 | 0.2151 |
| 8 | 0 | 9, 7, 7, 8 | no | 0.9132 | 0.6547 |
| 9 | 0 | 3, 7, 7, 2 | no | 0.9953 | 0.9819 |
| 10 | 1 | 3, 0, 7, 3 | no | 0.9857 | 0.9981 |
| 11 | 1 | 1, 9, 2, 5 | no | 0.9492 | 0.8218 |
| 12 | 1 | 2, 2, 2, 4 | no | 0.9729 | 0.6482 |
| 13 | 1 | 2, 3, 2, 5 | no | 0.9725 | 0.8603 |
| 14 | 1 | 1, 7, 2, 6 | no | 0.9919 | 1.0366 |
| 15 | 1 | 6, 1, 1, 6 | no | 0.9997 | 1.0165 |
| 16 | 1 | 5, 9, 0, 4 | no | 0.9398 | 0.9733 |
| 17 | 1 | 5, 4, 2, 8 | no | -0.1549 | 0.2888 |
| 18 | 1 | 2, 9, 0, 8 | no | 0.9682 | 0.7470 |
| 19 | 1 | 1, 4, 2, 3 | no | 0.9578 | 0.6398 |
| 20 | 2 | 8, 4, 0, 2 | no | 0.1209 | 0.3980 |
| 21 | 2 | 9, 9, 6, 6 | no | 0.9971 | 0.9721 |
| 22 | 2 | 1, 7, 4, 1 | no | 0.9960 | 0.5671 |
| 23 | 2 | 8, 1, 0, 7 | no | 0.9704 | 0.5816 |
| 24 | 2 | 2, 5, 1, 1 | no | 0.9888 | 0.9256 |
| 25 | 2 | 0, 7, 0, 1 | no | -0.0289 | 0.2268 |
| 26 | 2 | 4, 4, 0, 5 | no | 0.0400 | 0.2958 |
| 27 | 2 | 5, 8, 0, 4 | no | -0.1297 | 0.1971 |
| 28 | 2 | 6, 3, 5, 5 | no | 0.0843 | 0.1744 |
| 29 | 2 | 9, 8, 2, 3 | no | -0.0274 | 0.0508 |

## 9. Releases per cell

Expected: every release in the baseline file for that N. No release is dropped after a result is seen.

| Setting | N | E | Split | Attacker | Releases | Of | Missing release ids |
|---|---|---|---|---|---|---|---|
| mnist-fresh | 2 | 1 | 1x2 | attacker 1, basic | 30 | 30 | none |
| mnist-fresh | 2 | 1 | 1x2 | attacker 1, cosine | 30 | 30 | none |
| mnist-fresh | 2 | 5 | 1x2 | attacker 1, basic | 30 | 30 | none |
| mnist-fresh | 2 | 5 | 1x2 | attacker 1, cosine | 30 | 30 | none |
| mnist-fresh | 2 | 5 | 1x2 | attacker 2, cosine | 30 | 30 | none |
| mnist-fresh | 2 | 5 | 2x1 | attacker 1, basic | 30 | 30 | none |
| mnist-fresh | 2 | 5 | 2x1 | attacker 1, cosine | 30 | 30 | none |
| mnist-fresh | 2 | 5 | 2x1 | attacker 2, cosine | 30 | 30 | none |
| mnist-trained | 2 | 1 | 1x2 | attacker 1, basic | 30 | 30 | none |
| mnist-trained | 2 | 1 | 1x2 | attacker 1, cosine | 30 | 30 | none |
| mnist-trained | 2 | 5 | 1x2 | attacker 1, basic | 30 | 30 | none |
| mnist-trained | 2 | 5 | 1x2 | attacker 1, cosine | 30 | 30 | none |
| mnist-trained | 2 | 5 | 1x2 | attacker 2, cosine | 30 | 30 | none |
| mnist-trained | 2 | 5 | 2x1 | attacker 1, basic | 30 | 30 | none |
| mnist-trained | 2 | 5 | 2x1 | attacker 1, cosine | 30 | 30 | none |
| mnist-trained | 2 | 5 | 2x1 | attacker 2, cosine | 30 | 30 | none |
| mnist-fresh | 4 | 1 | 1x4 | attacker 1, basic | 30 | 30 | none |
| mnist-fresh | 4 | 1 | 1x4 | attacker 1, cosine | 30 | 30 | none |
| mnist-fresh | 4 | 5 | 1x4 | attacker 1, basic | 30 | 30 | none |
| mnist-fresh | 4 | 5 | 1x4 | attacker 1, cosine | 30 | 30 | none |
| mnist-fresh | 4 | 5 | 1x4 | attacker 2, cosine | 30 | 30 | none |
| mnist-fresh | 4 | 5 | 4x1 | attacker 1, basic | 30 | 30 | none |
| mnist-fresh | 4 | 5 | 4x1 | attacker 1, cosine | 30 | 30 | none |
| mnist-fresh | 4 | 5 | 4x1 | attacker 2, cosine | 30 | 30 | none |
| mnist-trained | 4 | 1 | 1x4 | attacker 1, basic | 30 | 30 | none |
| mnist-trained | 4 | 1 | 1x4 | attacker 1, cosine | 30 | 30 | none |
| mnist-trained | 4 | 5 | 1x4 | attacker 1, basic | 30 | 30 | none |
| mnist-trained | 4 | 5 | 1x4 | attacker 1, cosine | 30 | 30 | none |
| mnist-trained | 4 | 5 | 1x4 | attacker 2, cosine | 30 | 30 | none |
| mnist-trained | 4 | 5 | 4x1 | attacker 1, basic | 30 | 30 | none |
| mnist-trained | 4 | 5 | 4x1 | attacker 1, cosine | 30 | 30 | none |
| mnist-trained | 4 | 5 | 4x1 | attacker 2, cosine | 30 | 30 | none |

Settings (fresh vs trained) use the same releases; any comparison between them is descriptive.
