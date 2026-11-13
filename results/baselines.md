# No-information baselines (MNIST test images 50-81)

A reconstruction must score clearly above these to show a leak. ± is the 95% CI over seeds (deterministic rows have none).

| Fake reconstruction | SSIM | PSNR (dB) | MSE |
|---|---|---|---|
| all black | 0.421 | 10.108 | 0.104 |
| average digit | 0.336 | 11.905 | 0.067 |
| random noise | 0.002 ± 0.003 | 5.018 ± 0.020 | 0.315 ± 0.002 |
| unrelated digits, best-matched, K=1 | 0.368 ± 0.014 | 8.905 ± 0.142 | 0.134 ± 0.003 |
| unrelated digits, best-matched, K=2 | 0.395 ± 0.016 | 9.167 ± 0.247 | 0.128 ± 0.005 |
| unrelated digits, best-matched, K=4 | 0.415 ± 0.023 | 9.394 ± 0.283 | 0.121 ± 0.005 |
| unrelated digits, best-matched, K=8 | 0.473 ± 0.026 | 10.064 ± 0.270 | 0.105 ± 0.004 |
| unrelated digits, best-matched, K=16 | 0.494 ± 0.022 | 10.383 ± 0.446 | 0.099 ± 0.008 |
