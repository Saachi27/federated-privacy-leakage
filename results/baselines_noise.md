# No-information baselines (MNIST test images 50-65)

A reconstruction must score clearly above the **label-aware baseline** to show a leak, because the attackers are given the labels. SSIM is the standard form; the old zero-padded SSIM is shown for comparison. ± is the 95% CI over seeds.

| Fake reconstruction | SSIM | Old SSIM (zero-padded) | PSNR (dB) | MSE |
|---|---|---|---|---|
| all black | 0.100 | 0.428 | 10.445 | 0.098 |
| average digit | 0.103 | 0.332 | 12.037 | 0.065 |
| class average (knows labels) | 0.232 | 0.443 | 12.689 | 0.058 |
| random noise | -0.002 ± 0.012 | 0.001 ± 0.006 | 5.023 ± 0.037 | 0.315 ± 0.003 |
| unrelated digits, best-matched, K=1 | 0.087 ± 0.030 | 0.353 ± 0.029 | 8.768 ± 0.397 | 0.138 ± 0.012 |
| same-class digits, best-matched, K=1 (knows labels) | 0.190 ± 0.050 | 0.427 ± 0.042 | 9.957 ± 0.486 | 0.111 ± 0.008 |
| label-aware baseline, K=1 | 0.236 ± 0.010 | 0.448 ± 0.014 | 12.689 | 0.058 |
| unrelated digits, best-matched, K=2 | 0.146 ± 0.034 | 0.397 ± 0.016 | 9.336 ± 0.314 | 0.123 ± 0.006 |
| same-class digits, best-matched, K=2 (knows labels) | 0.208 ± 0.033 | 0.440 ± 0.031 | 9.844 ± 0.322 | 0.111 ± 0.009 |
| label-aware baseline, K=2 | 0.232 | 0.451 ± 0.018 | 12.689 | 0.058 |
| unrelated digits, best-matched, K=4 | 0.177 ± 0.015 | 0.413 ± 0.010 | 9.493 ± 0.202 | 0.119 ± 0.001 |
| same-class digits, best-matched, K=4 (knows labels) | 0.264 ± 0.044 | 0.475 ± 0.035 | 10.283 ± 0.382 | 0.101 ± 0.007 |
| label-aware baseline, K=4 | 0.269 ± 0.033 | 0.477 ± 0.030 | 12.689 | 0.058 |
| unrelated digits, best-matched, K=8 | 0.236 ± 0.026 | 0.458 ± 0.021 | 9.986 ± 0.272 | 0.105 ± 0.006 |
| same-class digits, best-matched, K=8 (knows labels) | 0.289 ± 0.026 | 0.498 ± 0.017 | 10.534 ± 0.141 | 0.095 ± 0.004 |
| label-aware baseline, K=8 | 0.289 ± 0.026 | 0.498 ± 0.017 | 12.689 | 0.058 |
