# Attack on the sum over K examples (mnist-fresh, 3 seeds)

| K | Attacker | Sums | Images | Mean SSIM (95% CI over seeds) | Success (SSIM≥0.9) | Best image per sum |
|---|---|---|---|---|---|---|
| 8 | basic | 12 | 96 | 0.792 ± 0.060 | 42% | 0.971 |
| 8 | cosine | 12 | 96 | 0.647 ± 0.106 | 17% | 0.915 |
| 16 | basic | 6 | 96 | 0.278 ± 0.155 | 0% | 0.562 |
| 16 | cosine | 6 | 96 | 0.443 ± 0.160 | 1% | 0.819 |
