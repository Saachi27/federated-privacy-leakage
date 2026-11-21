# Pilot results (tuning images only; not a study result)

*Made by `scripts/analyze_journal.py`. Main objective of attacker 1: **cosine**. Attacker 2 uses cosine.*

## 1. Main endpoint

MNIST fresh, N = 2, E = 5, attacker 2 (cosine). Positive control: attacker 1, main objective, E = 1.

**Outcome: more**

| Part | Estimate [interval] | Rule | Result |
|---|---|---|---|
| Positive control (s − b, E = 1) | 0.536 [0.446, 0.627] (95%) | lower end > 0 | passes |
| 1 × 2 beats baseline (s − b) | -0.013 [-0.119, 0.093] (95%) | lower end > 0 | no |
| 2 × 1 beats baseline (s − b) | 0.538 [0.446, 0.629] (95%) | lower end > 0 | yes |
| Gap 2 × 1 − 1 × 2 | 0.551 [0.463, 0.638] (90%) | margin ±0.05 | n = 25 |

Bootstrap check (report only): gap 90% 0.551 [0.472, 0.626]; control 95% 0.536 [0.452, 0.614].

Sentence: "Under attacker 2, on MNIST fresh at N = 2 and E = 5, the 2-client sum gave better reconstructions than the 1-client update (gap 0.551, 90% interval [0.463, 0.638], 25 paired releases on 3 checkpoints)."

## 2. Secondary labels (E = 5)

Same rules, each (setting, N) with its own positive control (attacker 1, main objective, E = 1). Not adjusted for multiple tests.

| Setting | N | Attacker | Outcome | PC s − b (95%) | 1×N s − b (95%) | N×1 s − b (95%) | Gap (90%) |
|---|---|---|---|---|---|---|---|
| mnist-fresh | 2 | attacker 2, cosine | more | 0.536 [0.446, 0.627] | -0.013 [-0.119, 0.093] | 0.538 [0.446, 0.629] | 0.551 [0.463, 0.638] |
| mnist-fresh | 2 | attacker 1, cosine | more | 0.536 [0.446, 0.627] | -0.213 [-0.304, -0.123] | 0.536 [0.446, 0.627] | 0.750 [0.667, 0.832] |
| mnist-fresh | 2 | attacker 1, basic | neither above baseline | 0.536 [0.446, 0.627] | -0.291 [-0.335, -0.247] | -0.275 [-0.326, -0.223] | 0.016 [-0.001, 0.034] |
| mnist-trained | 2 | attacker 2, cosine | descriptive only | 0.183 [–, –] | -0.195 [–, –] | 0.346 [–, –] | 0.541 [–, –] |
| mnist-trained | 2 | attacker 1, cosine | descriptive only | 0.183 [–, –] | 0.217 [–, –] | 0.243 [–, –] | 0.027 [–, –] |
| mnist-trained | 2 | attacker 1, basic | descriptive only | 0.183 [–, –] | -0.336 [–, –] | -0.123 [–, –] | 0.213 [–, –] |
| mnist-fresh | 4 | attacker 2, cosine | descriptive only | 0.433 [-0.599, 1.465] | -0.259 [-0.300, -0.219] | 0.429 [-0.667, 1.525] | 0.688 [0.124, 1.253] |
| mnist-fresh | 4 | attacker 1, cosine | descriptive only | 0.433 [-0.599, 1.465] | -0.291 [-0.319, -0.263] | 0.438 [-0.612, 1.487] | 0.729 [0.221, 1.237] |
| mnist-fresh | 4 | attacker 1, basic | descriptive only | 0.433 [-0.599, 1.465] | -0.290 [-0.307, -0.273] | -0.280 [-0.397, -0.163] | 0.010 [-0.040, 0.059] |

## 3. Second endpoint: any image passes (SSIM ≥ 0.9)

| Setting | N | E | Split | Attacker | Rate | Rate − baseline (95%) |
|---|---|---|---|---|---|---|
| mnist-fresh | 2 | 1 | 1x2 | attacker 1, cosine | 0.880 | 0.864 [0.723, 1.005] |
| mnist-fresh | 2 | 1 | 1x2 | attacker 1, basic | 1.000 | 0.984 [0.959, 1.009] |
| mnist-fresh | 2 | 5 | 1x2 | attacker 1, cosine | 0.040 | 0.024 [-0.069, 0.117] |
| mnist-fresh | 2 | 5 | 1x2 | attacker 1, basic | 0.000 | -0.016 [-0.041, 0.009] |
| mnist-fresh | 2 | 5 | 1x2 | attacker 2, cosine | 0.040 | 0.024 [-0.069, 0.117] |
| mnist-fresh | 2 | 5 | 2x1 | attacker 1, cosine | 0.880 | 0.864 [0.723, 1.005] |
| mnist-fresh | 2 | 5 | 2x1 | attacker 1, basic | 0.000 | -0.016 [-0.041, 0.009] |
| mnist-fresh | 2 | 5 | 2x1 | attacker 2, cosine | 0.880 | 0.864 [0.723, 1.005] |
| mnist-fresh | 4 | 1 | 1x4 | attacker 1, cosine | 1.000 | 1.000 [1.000, 1.000] |
| mnist-fresh | 4 | 1 | 1x4 | attacker 1, basic | 1.000 | 1.000 [1.000, 1.000] |
| mnist-fresh | 4 | 5 | 1x4 | attacker 1, cosine | 0.000 | 0.000 [0.000, 0.000] |
| mnist-fresh | 4 | 5 | 1x4 | attacker 1, basic | 0.000 | 0.000 [0.000, 0.000] |
| mnist-fresh | 4 | 5 | 1x4 | attacker 2, cosine | 0.000 | 0.000 [0.000, 0.000] |
| mnist-fresh | 4 | 5 | 4x1 | attacker 1, cosine | 1.000 | 1.000 [1.000, 1.000] |
| mnist-fresh | 4 | 5 | 4x1 | attacker 1, basic | 0.000 | 0.000 [0.000, 0.000] |
| mnist-fresh | 4 | 5 | 4x1 | attacker 2, cosine | 1.000 | 1.000 [1.000, 1.000] |
| mnist-trained | 2 | 1 | 1x2 | attacker 1, cosine | 0.000 | 0.000 [–, –] |
| mnist-trained | 2 | 1 | 1x2 | attacker 1, basic | 0.000 | 0.000 [–, –] |
| mnist-trained | 2 | 5 | 1x2 | attacker 1, cosine | 0.000 | 0.000 [–, –] |
| mnist-trained | 2 | 5 | 1x2 | attacker 1, basic | 0.000 | 0.000 [–, –] |
| mnist-trained | 2 | 5 | 1x2 | attacker 2, cosine | 0.000 | 0.000 [–, –] |
| mnist-trained | 2 | 5 | 2x1 | attacker 1, cosine | 0.000 | 0.000 [–, –] |
| mnist-trained | 2 | 5 | 2x1 | attacker 1, basic | 0.000 | 0.000 [–, –] |
| mnist-trained | 2 | 5 | 2x1 | attacker 2, cosine | 0.500 | 0.500 [–, –] |

| Setting | N | Attacker | Gap N×1 − 1×N (90%) | (95%) | Both / only N×1 / only 1×N / neither |
|---|---|---|---|---|---|
| mnist-fresh | 2 | attacker 1, cosine | 0.840 [0.713, 0.967] | 0.840 [0.685, 0.995] | 1 / 21 / 0 / 3 |
| mnist-fresh | 2 | attacker 1, basic | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0 / 0 / 0 / 25 |
| mnist-fresh | 2 | attacker 2, cosine | 0.840 [0.713, 0.967] | 0.840 [0.685, 0.995] | 1 / 21 / 0 / 3 |
| mnist-fresh | 4 | attacker 1, cosine | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 / 4 / 0 / 0 |
| mnist-fresh | 4 | attacker 1, basic | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0 / 0 / 0 / 4 |
| mnist-fresh | 4 | attacker 2, cosine | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 / 4 / 0 / 0 |
| mnist-trained | 2 | attacker 1, cosine | 0.000 [–, –] | 0.000 [–, –] | 0 / 0 / 0 / 2 |
| mnist-trained | 2 | attacker 1, basic | 0.000 [–, –] | 0.000 [–, –] | 0 / 0 / 0 / 2 |
| mnist-trained | 2 | attacker 2, cosine | 0.500 [–, –] | 0.500 [–, –] | 0 / 1 / 0 / 1 |

## 4. Averages hide the risk

| Setting | N | E | Split | Attacker | Releases | Mean SSIM | Any passes | Retrieval hit | SSIM per checkpoint |
|---|---|---|---|---|---|---|---|---|---|
| mnist-fresh | 2 | 1 | 1x2 | attacker 1, cosine | 25 | 0.856 | 0.880 | 1.000 | k0: 0.864, k1: 0.831, k2: 0.873 |
| mnist-fresh | 2 | 1 | 1x2 | attacker 1, basic | 25 | 0.978 | 1.000 | 1.000 | k0: 0.998, k1: 0.932, k2: 1.000 |
| mnist-fresh | 2 | 5 | 1x2 | attacker 1, cosine | 25 | 0.107 | 0.040 | 0.160 | k0: 0.163, k1: 0.066, k2: 0.084 |
| mnist-fresh | 2 | 5 | 1x2 | attacker 1, basic | 25 | 0.029 | 0.000 | 0.000 | k0: 0.031, k1: 0.024, k2: 0.032 |
| mnist-fresh | 2 | 5 | 1x2 | attacker 2, cosine | 25 | 0.307 | 0.040 | 0.920 | k0: 0.276, k1: 0.229, k2: 0.419 |
| mnist-fresh | 2 | 5 | 2x1 | attacker 1, cosine | 25 | 0.856 | 0.880 | 1.000 | k0: 0.861, k1: 0.830, k2: 0.877 |
| mnist-fresh | 2 | 5 | 2x1 | attacker 1, basic | 25 | 0.045 | 0.000 | 0.080 | k0: 0.059, k1: 0.031, k2: 0.045 |
| mnist-fresh | 2 | 5 | 2x1 | attacker 2, cosine | 25 | 0.858 | 0.880 | 1.000 | k0: 0.870, k1: 0.831, k2: 0.870 |
| mnist-fresh | 4 | 1 | 1x4 | attacker 1, cosine | 4 | 0.763 | 1.000 | 1.000 | k0: 0.741, k1: 0.613, k2: 0.958 |
| mnist-fresh | 4 | 1 | 1x4 | attacker 1, basic | 4 | 0.924 | 1.000 | 1.000 | k0: 0.979, k1: 0.739, k2: 1.000 |
| mnist-fresh | 4 | 5 | 1x4 | attacker 1, cosine | 4 | 0.039 | 0.000 | 0.000 | k0: 0.031, k1: 0.026, k2: 0.069 |
| mnist-fresh | 4 | 5 | 1x4 | attacker 1, basic | 4 | 0.040 | 0.000 | 0.000 | k0: 0.037, k1: 0.026, k2: 0.062 |
| mnist-fresh | 4 | 5 | 1x4 | attacker 2, cosine | 4 | 0.071 | 0.000 | 0.250 | k0: 0.059, k1: 0.107, k2: 0.058 |
| mnist-fresh | 4 | 5 | 4x1 | attacker 1, cosine | 4 | 0.768 | 1.000 | 1.000 | k0: 0.751, k1: 0.618, k2: 0.953 |
| mnist-fresh | 4 | 5 | 4x1 | attacker 1, basic | 4 | 0.050 | 0.000 | 0.000 | k0: 0.041, k1: 0.038, k2: 0.080 |
| mnist-fresh | 4 | 5 | 4x1 | attacker 2, cosine | 4 | 0.759 | 1.000 | 1.000 | k0: 0.732, k1: 0.618, k2: 0.955 |
| mnist-trained | 2 | 1 | 1x2 | attacker 1, cosine | 2 | 0.566 | 0.000 | 1.000 | k0: 0.541, k1: 0.590 |
| mnist-trained | 2 | 1 | 1x2 | attacker 1, basic | 2 | 0.301 | 0.000 | 0.500 | k0: -0.008, k1: 0.610 |
| mnist-trained | 2 | 5 | 1x2 | attacker 1, cosine | 2 | 0.599 | 0.000 | 1.000 | k0: 0.547, k1: 0.651 |
| mnist-trained | 2 | 5 | 1x2 | attacker 1, basic | 2 | 0.047 | 0.000 | 0.000 | k0: 0.017, k1: 0.077 |
| mnist-trained | 2 | 5 | 1x2 | attacker 2, cosine | 2 | 0.188 | 0.000 | 0.500 | k0: 0.234, k1: 0.141 |
| mnist-trained | 2 | 5 | 2x1 | attacker 1, cosine | 2 | 0.625 | 0.000 | 1.000 | k0: 0.573, k1: 0.678 |
| mnist-trained | 2 | 5 | 2x1 | attacker 1, basic | 2 | 0.259 | 0.000 | 0.500 | k0: 0.089, k1: 0.429 |
| mnist-trained | 2 | 5 | 2x1 | attacker 2, cosine | 2 | 0.729 | 0.500 | 1.000 | k0: 0.665, k1: 0.792 |
| label-aware baseline | 2 | – | – | max(class average, same class) | 25 | 0.320 | 0.016 | 0.008 | – |
| label-aware baseline | 4 | – | – | max(class average, same class) | 4 | 0.330 | 0.000 | 0.000 | – |

Baseline parts and context (mean SSIM): N = 2: class average 0.306, same class 0.272, exact retrieval chance 0.0021, all black 0.119, average digit 0.103, random noise 0.013; N = 4: class average 0.325, same class 0.290, exact retrieval chance 0.0045, all black 0.108, average digit 0.100, random noise 0.024

## 5. Drift (E = 5)

| Setting | N | Releases | Median | Range | Median per checkpoint | Median norm ratio | E = 1 drift, min |
|---|---|---|---|---|---|---|---|
| mnist-fresh | 2 | 25 | 0.8197 | 0.2422–1.0000 | k0: 0.820, k1: 0.836, k2: 0.809 | 0.8383 | 1.000000 |
| mnist-trained | 2 | 2 | 0.9995 | 0.9993–0.9996 | k0: 1.000, k1: 0.999 | 0.7701 | 1.000000 |
| mnist-fresh | 4 | 4 | 0.5337 | 0.2943–0.7938 | k0: 0.481, k1: 0.400, k2: 0.794 | 0.7329 | 1.000000 |

## 6. Secondary objective (attacker 1, basic)

| Setting | N | E | Split | Mean SSIM | s − b (95%) | Any passes |
|---|---|---|---|---|---|---|
| mnist-fresh | 2 | 1 | 1x2 | 0.978 | 0.658 [0.609, 0.706] | 1.000 |
| mnist-fresh | 2 | 5 | 1x2 | 0.029 | -0.291 [-0.335, -0.247] | 0.000 |
| mnist-fresh | 2 | 5 | 2x1 | 0.045 | -0.275 [-0.326, -0.223] | 0.000 |
| mnist-fresh | 4 | 1 | 1x4 | 0.924 | 0.594 [0.485, 0.702] | 1.000 |
| mnist-fresh | 4 | 5 | 1x4 | 0.040 | -0.290 [-0.307, -0.273] | 0.000 |
| mnist-fresh | 4 | 5 | 4x1 | 0.050 | -0.280 [-0.397, -0.163] | 0.000 |
| mnist-trained | 2 | 1 | 1x2 | 0.301 | -0.081 [–, –] | 0.000 |
| mnist-trained | 2 | 5 | 1x2 | 0.047 | -0.336 [–, –] | 0.000 |
| mnist-trained | 2 | 5 | 2x1 | 0.259 | -0.123 [–, –] | 0.000 |

## 7. Retrieval (top-1 in the slot label's class gallery)

| Setting | N | E | Split | Attacker | Hit rate | Hit − baseline (95%) | Per checkpoint |
|---|---|---|---|---|---|---|---|
| mnist-fresh | 2 | 1 | 1x2 | attacker 1, cosine | 1.000 | 0.992 [0.973, 1.011] | k0: 1.000, k1: 1.000, k2: 1.000 |
| mnist-fresh | 2 | 1 | 1x2 | attacker 1, basic | 1.000 | 0.992 [0.973, 1.011] | k0: 1.000, k1: 1.000, k2: 1.000 |
| mnist-fresh | 2 | 5 | 1x2 | attacker 1, cosine | 0.160 | 0.152 [-0.011, 0.315] | k0: 0.222, k1: 0.125, k2: 0.125 |
| mnist-fresh | 2 | 5 | 1x2 | attacker 1, basic | 0.000 | -0.008 [-0.027, 0.011] | k0: 0.000, k1: 0.000, k2: 0.000 |
| mnist-fresh | 2 | 5 | 1x2 | attacker 2, cosine | 0.920 | 0.912 [0.791, 1.033] | k0: 1.000, k1: 0.875, k2: 0.875 |
| mnist-fresh | 2 | 5 | 2x1 | attacker 1, cosine | 1.000 | 0.992 [0.973, 1.011] | k0: 1.000, k1: 1.000, k2: 1.000 |
| mnist-fresh | 2 | 5 | 2x1 | attacker 1, basic | 0.080 | 0.072 [-0.052, 0.196] | k0: 0.111, k1: 0.000, k2: 0.125 |
| mnist-fresh | 2 | 5 | 2x1 | attacker 2, cosine | 1.000 | 0.992 [0.973, 1.011] | k0: 1.000, k1: 1.000, k2: 1.000 |
| mnist-fresh | 4 | 1 | 1x4 | attacker 1, cosine | 1.000 | 1.000 [1.000, 1.000] | k0: 1.000, k1: 1.000, k2: 1.000 |
| mnist-fresh | 4 | 1 | 1x4 | attacker 1, basic | 1.000 | 1.000 [1.000, 1.000] | k0: 1.000, k1: 1.000, k2: 1.000 |
| mnist-fresh | 4 | 5 | 1x4 | attacker 1, cosine | 0.000 | 0.000 [0.000, 0.000] | k0: 0.000, k1: 0.000, k2: 0.000 |
| mnist-fresh | 4 | 5 | 1x4 | attacker 1, basic | 0.000 | 0.000 [0.000, 0.000] | k0: 0.000, k1: 0.000, k2: 0.000 |
| mnist-fresh | 4 | 5 | 1x4 | attacker 2, cosine | 0.250 | 0.250 [-2.927, 3.427] | k0: 0.500, k1: 0.000, k2: 0.000 |
| mnist-fresh | 4 | 5 | 4x1 | attacker 1, cosine | 1.000 | 1.000 [1.000, 1.000] | k0: 1.000, k1: 1.000, k2: 1.000 |
| mnist-fresh | 4 | 5 | 4x1 | attacker 1, basic | 0.000 | 0.000 [0.000, 0.000] | k0: 0.000, k1: 0.000, k2: 0.000 |
| mnist-fresh | 4 | 5 | 4x1 | attacker 2, cosine | 1.000 | 1.000 [1.000, 1.000] | k0: 1.000, k1: 1.000, k2: 1.000 |
| mnist-trained | 2 | 1 | 1x2 | attacker 1, cosine | 1.000 | 1.000 [–, –] | k0: 1.000, k1: 1.000 |
| mnist-trained | 2 | 1 | 1x2 | attacker 1, basic | 0.500 | 0.500 [–, –] | k0: 0.000, k1: 1.000 |
| mnist-trained | 2 | 5 | 1x2 | attacker 1, cosine | 1.000 | 1.000 [–, –] | k0: 1.000, k1: 1.000 |
| mnist-trained | 2 | 5 | 1x2 | attacker 1, basic | 0.000 | 0.000 [–, –] | k0: 0.000, k1: 0.000 |
| mnist-trained | 2 | 5 | 1x2 | attacker 2, cosine | 0.500 | 0.500 [–, –] | k0: 1.000, k1: 0.000 |
| mnist-trained | 2 | 5 | 2x1 | attacker 1, cosine | 1.000 | 1.000 [–, –] | k0: 1.000, k1: 1.000 |
| mnist-trained | 2 | 5 | 2x1 | attacker 1, basic | 0.500 | 0.500 [–, –] | k0: 0.000, k1: 1.000 |
| mnist-trained | 2 | 5 | 2x1 | attacker 2, cosine | 1.000 | 1.000 [–, –] | k0: 1.000, k1: 1.000 |

| Setting | N | Attacker | Hit gap N×1 − 1×N (90%) | (95%) | Both / only N×1 / only 1×N / neither |
|---|---|---|---|---|---|
| mnist-fresh | 2 | attacker 1, cosine | 0.840 [0.707, 0.973] | 0.840 [0.679, 1.001] | 4 / 21 / 0 / 0 |
| mnist-fresh | 2 | attacker 1, basic | 0.080 [-0.019, 0.179] | 0.080 [-0.041, 0.201] | 0 / 2 / 0 / 23 |
| mnist-fresh | 2 | attacker 2, cosine | 0.080 [-0.020, 0.180] | 0.080 [-0.041, 0.201] | 23 / 2 / 0 / 0 |
| mnist-fresh | 4 | attacker 1, cosine | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 / 4 / 0 / 0 |
| mnist-fresh | 4 | attacker 1, basic | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0 / 0 / 0 / 4 |
| mnist-fresh | 4 | attacker 2, cosine | 0.750 [-0.828, 2.328] | 0.750 [-2.427, 3.927] | 1 / 3 / 0 / 0 |
| mnist-trained | 2 | attacker 1, cosine | 0.000 [–, –] | 0.000 [–, –] | 2 / 0 / 0 / 0 |
| mnist-trained | 2 | attacker 1, basic | 0.500 [–, –] | 0.500 [–, –] | 0 / 1 / 0 / 1 |
| mnist-trained | 2 | attacker 2, cosine | 0.500 [–, –] | 0.500 [–, –] | 1 / 1 / 0 / 0 |

Settings (fresh vs trained) use the same releases; any comparison between them is descriptive.
