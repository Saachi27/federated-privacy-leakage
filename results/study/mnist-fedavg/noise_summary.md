# Noise on the sum over K examples (mnist-fedavg)

Clip C = 32.86373157501221. Epsilon is for one Gaussian release (delta=1e-5, sampling rate 1, one step); it is not the DP-SGD training epsilon.

| K | sigma | epsilon | Attacker | Images | Mean SSIM (95% CI over seeds) | Success (SSIM≥0.9) | Mean PSNR | Clipped |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | ∞ | basic | 96 | 0.189 ± 0.130 | 1% | 6.4 | 0% |
| 1 | 0 | ∞ | cosine | 96 | 0.650 ± 0.097 | 15% | 16.2 | 0% |
| 2 | 0 | ∞ | basic | 96 | 0.147 ± 0.207 | 0% | 4.6 | 0% |
| 2 | 0 | ∞ | cosine | 96 | 0.403 ± 0.245 | 0% | 9.3 | 0% |
| 4 | 0 | ∞ | basic | 96 | 0.129 ± 0.143 | 0% | 4.5 | 0% |
| 4 | 0 | ∞ | cosine | 96 | 0.302 ± 0.249 | 0% | 8.1 | 0% |
| 8 | 0 | ∞ | basic | 96 | 0.087 ± 0.057 | 0% | 3.8 | 0% |
| 8 | 0 | ∞ | cosine | 96 | 0.237 ± 0.209 | 0% | 7.5 | 0% |
