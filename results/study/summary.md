# Study Summary: Four Settings

Made by `scripts/compare_settings.py` from `results/study/`. Each setting has its own full report in `results/study/<setting>/rq4_summary.md`.

**How to read this**
- The attacker sees the **sum** of K images' updates, with or without noise.
- A score only counts as a **leak** if it is clearly above what a no-information guess scores. The guess knows the labels, like the attacker does (the **label-aware baseline**).
- "Clearly" means the whole 95% interval over seeds is above that line.
- SSIM is the standard form. PSNR is a second check.

## The Settings

| Setting | What is attacked | Clip C | Cosine TV | Model accuracy | Seeds |
|---|---|---|---|---|---|
| mnist-fresh | MNIST, fresh LeNet, one FedSGD step | 32.9 | 0.001 | fresh (untrained) | [0, 1, 2] |
| mnist-trained | MNIST, LeNet trained with FedAvg for 20 rounds, one FedSGD step | 27.1 | 0.01 | 0.937, 0.938, 0.936 | [0, 1, 2] |
| cifar10-fresh | CIFAR-10, fresh LeNet, one FedSGD step | 38.9 | 0.01 | fresh (untrained) | [0, 1, 2] |
| mnist-fedavg | MNIST, fresh LeNet, real FedAvg update (5 local steps at lr 0.01) | — (no noise) | 0.003 | fresh (untrained) | [0, 1, 2] |

Clip C and the cosine TV weight were chosen on held-out images (`scripts/tune_attack.py`). C sits just above the largest per-example gradient norm, so clipping almost never acts and the noise sweep measures noise only.

## 1. Does the Sum Leak? (No Noise)

Each cell: verdict from SSIM, then SSIM minus baseline / PSNR minus baseline. ✓ = PSNR gives the same verdict, ✗ = it does not. "At or below line" is not proof that nothing leaked: averages hide single images that do leak.

| Setting | Attacker | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|---|
| MNIST, fresh | basic | **leak** +0.73 / +45.6 dB ✓ | **leak** +0.70 / +40.4 dB ✓ | **leak** +0.67 / +28.6 dB ✓ | **leak** +0.49 / +11.2 dB ✓ | no clear leak -0.06 / -5.0 dB ✗ |
| MNIST, fresh | cosine | **leak** +0.72 / +23.5 dB ✓ | **leak** +0.59 / +14.9 dB ✓ | **leak** +0.46 / +8.4 dB ✓ | **leak** +0.26 / +3.1 dB ✓ | no clear leak +0.04 / -0.1 dB ✓ |
| MNIST, trained | basic | no clear leak -0.16 / -5.7 dB ✓ | no clear leak -0.15 / -6.4 dB ✓ | no clear leak +0.02 / -3.0 dB ✓ | no clear leak -0.02 / -4.2 dB ✗ | *at or below line* -0.14 / -6.3 dB ✓ |
| MNIST, trained | cosine | **leak** +0.69 / +15.2 dB ✓ | **leak** +0.35 / +5.2 dB ✓ | **leak** +0.17 / +0.8 dB ✗ | no clear leak -0.02 / -3.2 dB ✗ | *at or below line* -0.13 / -4.9 dB ✓ |
| CIFAR-10, fresh | basic | no clear leak +0.48 / +20.5 dB ✓ | no clear leak -0.02 / -3.8 dB ✓ | *at or below line* -0.11 / -7.1 dB ✓ | *at or below line* -0.12 / -7.2 dB ✓ | *at or below line* -0.13 / -7.1 dB ✓ |
| CIFAR-10, fresh | cosine | **leak** +0.62 / +8.7 dB ✓ | **leak** +0.42 / +5.0 dB ✗ | no clear leak +0.26 / +2.4 dB ✓ | no clear leak +0.14 / +0.8 dB ✓ | no clear leak +0.08 / +0.3 dB ✓ |
| MNIST, FedAvg update | basic | no clear leak -0.08 / -6.4 dB ✗ | no clear leak -0.12 / -8.2 dB ✗ | no clear leak -0.14 / -8.3 dB ✗ | *at or below line* -0.21 / -9.0 dB ✓ | *at or below line* -0.26 / -9.2 dB ✓ |
| MNIST, FedAvg update | cosine | **leak** +0.38 / +3.4 dB ✗ | no clear leak +0.14 / -3.4 dB ✓ | no clear leak +0.04 / -4.7 dB ✗ | no clear leak -0.06 / -5.3 dB ✗ | *at or below line* -0.15 / -5.5 dB ✓ |

## 2. How Much Noise Stops the Attack?

The lowest tested noise σ from which the attack is stopped (the whole interval at or below the baseline) at that σ and every larger one. ε is for **one** release of the noisy sum (δ = 1e-5); it is not the training ε. "Not stopped" means even the largest tested σ did not stop it.

| Setting | Attacker | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|---|
| MNIST, fresh | basic | 0.003 (ε 6.1e+04) | 0.003 (ε 6.1e+04) | 0.001 (ε 5.5e+05) | 0.001 (ε 5.5e+05) |
| MNIST, fresh | cosine | 0.01 (ε 5.6e+03) | 0.01 (ε 5.6e+03) | 0.01 (ε 5.6e+03) | 0.01 (ε 5.6e+03) |
| MNIST, trained | basic | 1e-4 (ε 5.5e+07) | 1e-4 (ε 5.5e+07) | 3e-4 (ε 6.1e+06) | 3e-4 (ε 6.1e+06) |
| MNIST, trained | cosine | 0.003 (ε 6.1e+04) | 0.003 (ε 6.1e+04) | 0.003 (ε 6.1e+04) | 0.003 (ε 6.1e+04) |
| CIFAR-10, fresh | basic | 0.001 (ε 5.5e+05) | 1e-4 (ε 5.5e+07) | 0 (no noise needed) | 0 (no noise needed) |
| CIFAR-10, fresh | cosine | 0.01 (ε 5.6e+03) | 0.01 (ε 5.6e+03) | 0.01 (ε 5.6e+03) | 0.01 (ε 5.6e+03) |

PSNR cross-check of the same rule:

| Setting | Attacker | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|---|
| MNIST, fresh | basic | 0.003 (ε 6.1e+04) | 0.001 (ε 5.5e+05) | 0.001 (ε 5.5e+05) | 3e-4 (ε 6.1e+06) |
| MNIST, fresh | cosine | 0.003 (ε 6.1e+04) | 0.003 (ε 6.1e+04) | 0.003 (ε 6.1e+04) | 0.003 (ε 6.1e+04) |
| MNIST, trained | basic | 1e-4 (ε 5.5e+07) | 1e-4 (ε 5.5e+07) | 1e-4 (ε 5.5e+07) | 0 (no noise needed) |
| MNIST, trained | cosine | 0.001 (ε 5.5e+05) | 0.001 (ε 5.5e+05) | 0.001 (ε 5.5e+05) | 0 (no noise needed) |
| CIFAR-10, fresh | basic | 3e-4 (ε 6.1e+06) | 1e-4 (ε 5.5e+07) | 0 (no noise needed) | 0 (no noise needed) |
| CIFAR-10, fresh | cosine | 0.003 (ε 6.1e+04) | 0.003 (ε 6.1e+04) | 0.01 (ε 5.6e+03) | 0.01 (ε 5.6e+03) |

## 3. The Sum vs an Attacker Who Knows the Other Images

Net = SSIM of the K-sum minus SSIM of an attacker who knows the other K−1 images (the K = 1 attack at the same σ). An optimal attacker can never do worse when it knows more, so net ≤ 0 is expected. What matters is **how big the gap is**. **Bold** = the 95% interval excludes 0.

| Setting | Attacker | K | Net, no noise | Net, σ = 0.001 |
|---|---|---|---|---|
| MNIST, fresh | basic | 2 | -0.030 ± 0.117 | **-0.210 ± 0.102** |
| MNIST, fresh | basic | 4 | -0.060 ± 0.123 | **-0.349 ± 0.064** |
| MNIST, fresh | basic | 8 | **-0.212 ± 0.068** | **-0.390 ± 0.082** |
| MNIST, fresh | cosine | 2 | **-0.128 ± 0.120** | **+0.005 ± 0.001** |
| MNIST, fresh | cosine | 4 | **-0.257 ± 0.111** | -0.036 ± 0.059 |
| MNIST, fresh | cosine | 8 | **-0.424 ± 0.037** | **-0.154 ± 0.049** |
| MNIST, trained | basic | 2 | +0.013 ± 0.057 | **+0.020 ± 0.019** |
| MNIST, trained | basic | 4 | +0.182 ± 0.363 | +0.048 ± 0.080 |
| MNIST, trained | basic | 8 | **+0.165 ± 0.140** | +0.058 ± 0.076 |
| MNIST, trained | cosine | 2 | **-0.334 ± 0.091** | -0.022 ± 0.051 |
| MNIST, trained | cosine | 4 | **-0.513 ± 0.082** | -0.041 ± 0.081 |
| MNIST, trained | cosine | 8 | **-0.671 ± 0.067** | **-0.098 ± 0.078** |
| CIFAR-10, fresh | basic | 2 | -0.491 ± 1.334 | +0.004 ± 0.036 |
| CIFAR-10, fresh | basic | 4 | -0.590 ± 1.330 | +0.001 ± 0.030 |
| CIFAR-10, fresh | basic | 8 | -0.596 ± 1.329 | -0.001 ± 0.022 |
| CIFAR-10, fresh | cosine | 2 | **-0.206 ± 0.200** | -0.063 ± 0.075 |
| CIFAR-10, fresh | cosine | 4 | **-0.365 ± 0.178** | **-0.167 ± 0.098** |
| CIFAR-10, fresh | cosine | 8 | **-0.481 ± 0.119** | **-0.261 ± 0.099** |

## 4. Did the Attack Just Run Out of Steps?

The same attack with 4× the steps, no noise.

| Setting | Attacker | K | Normal − baseline | 4× steps − baseline | 4× minus normal |
|---|---|---|---|---|---|
| MNIST, fresh | basic | 8 | +0.488 ± 0.142 | +0.496 ± 0.118 | +0.008 ± 0.043 |
| MNIST, fresh | cosine | 8 | +0.265 ± 0.040 | +0.351 ± 0.114 | +0.086 ± 0.141 |
| MNIST, fresh | basic | 16 | -0.064 ± 0.122 | -0.055 ± 0.135 | +0.009 ± 0.013 |
| MNIST, fresh | cosine | 16 | +0.036 ± 0.048 | +0.109 ± 0.113 | +0.073 ± 0.156 |

## 5. In Plain Words

- **MNIST, fresh, basic:** leaks clearly at K = 1, 2, 4, 8; no clear leak on average at K = 16. Noise σ ≥ 0.003 stops it at every tested K.
- **MNIST, fresh, cosine:** leaks clearly at K = 1, 2, 4, 8; no clear leak on average at K = 16. Noise σ ≥ 0.01 stops it at every tested K.
- **MNIST, trained, basic:** shows no clear leak at any K. Noise σ ≥ 3e-4 stops it at every tested K.
- **MNIST, trained, cosine:** leaks clearly at K = 1, 2, 4; no clear leak on average at K = 8, 16. Noise σ ≥ 0.003 stops it at every tested K.
- **CIFAR-10, fresh, basic:** shows no clear leak at any K. Noise σ ≥ 0.001 stops it at every tested K.
- **CIFAR-10, fresh, cosine:** leaks clearly at K = 1, 2; no clear leak on average at K = 4, 8, 16. Noise σ ≥ 0.01 stops it at every tested K.
- **MNIST, FedAvg update, basic:** shows no clear leak at any K.
- **MNIST, FedAvg update, cosine:** leaks clearly at K = 1; no clear leak on average at K = 2, 4, 8, 16.

- A low **average** score does not mean no image leaked. Look at the example images too: single images can be clearly recovered in a sum the average calls safe.
- The ε values that stop these attacks are very large. Stopping two practical attacks is not a privacy guarantee.

## Figures

- `results/fig_study_leak_vs_k.png`: leak vs K for every setting, no noise.
- `results/fig_study_stopping_noise.png`: stopping noise vs K, with ε.
- `results/study/<setting>/fig_rq4_heatmap.png`: leak over K and σ, per setting.

