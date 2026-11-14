# No-information baselines (cifar10 test images 50-81)

A reconstruction must score clearly above the **label-aware baseline** to show a leak, because the attackers are given the labels. SSIM is the standard form; the old zero-padded SSIM is shown for comparison. ± is the 95% CI over seeds.

| Fake reconstruction | SSIM | Old SSIM (zero-padded) | PSNR (dB) | MSE |
|---|---|---|---|---|
| all black | 0.002 | 0.009 | 6.150 | 0.273 |
| average digit | 0.143 | 0.349 | 12.534 | 0.064 |
| class average (knows labels) | 0.150 | 0.358 | 13.013 | 0.059 |
| random noise | 0.009 ± 0.003 | 0.129 ± 0.001 | 8.319 ± 0.013 | 0.151 ± 0.001 |
| unrelated digits, best-matched, K=1 | 0.051 ± 0.024 | 0.218 ± 0.014 | 9.437 ± 0.324 | 0.129 ± 0.011 |
| same-class digits, best-matched, K=1 (knows labels) | 0.059 ± 0.015 | 0.237 ± 0.015 | 9.952 ± 0.333 | 0.113 ± 0.010 |
| label-aware baseline, K=1 | 0.150 | 0.358 | 13.013 | 0.059 |
| unrelated digits, best-matched, K=2 | 0.067 ± 0.008 | 0.229 ± 0.009 | 9.510 ± 0.246 | 0.126 ± 0.005 |
| same-class digits, best-matched, K=2 (knows labels) | 0.084 ± 0.021 | 0.246 ± 0.017 | 10.050 ± 0.770 | 0.114 ± 0.020 |
| label-aware baseline, K=2 | 0.150 | 0.358 | 13.013 | 0.059 |
| unrelated digits, best-matched, K=4 | 0.095 ± 0.009 | 0.256 ± 0.013 | 10.033 ± 0.376 | 0.112 ± 0.011 |
| same-class digits, best-matched, K=4 (knows labels) | 0.106 ± 0.015 | 0.260 ± 0.017 | 10.126 ± 0.237 | 0.113 ± 0.006 |
| label-aware baseline, K=4 | 0.150 | 0.358 | 13.013 | 0.059 |
| unrelated digits, best-matched, K=8 | 0.124 ± 0.011 | 0.281 ± 0.014 | 10.505 ± 0.286 | 0.098 ± 0.006 |
| same-class digits, best-matched, K=8 (knows labels) | 0.128 ± 0.011 | 0.282 ± 0.011 | 10.537 ± 0.311 | 0.101 ± 0.012 |
| label-aware baseline, K=8 | 0.150 | 0.358 | 13.013 | 0.059 |
| unrelated digits, best-matched, K=16 | 0.151 ± 0.011 | 0.293 ± 0.016 | 10.653 ± 0.252 | 0.099 ± 0.006 |
| same-class digits, best-matched, K=16 (knows labels) | 0.147 ± 0.009 | 0.286 ± 0.003 | 10.513 ± 0.367 | 0.098 ± 0.008 |
| label-aware baseline, K=16 | 0.151 ± 0.002 | 0.358 | 13.013 | 0.059 |
