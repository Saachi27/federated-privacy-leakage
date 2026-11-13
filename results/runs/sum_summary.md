# Attack on the sum over K examples (MNIST, fresh LeNet, 5 seeds)

| K | Attacker | Sums | Images | Mean SSIM (95% CI over seeds) | Success (SSIM≥0.9) | Best image per sum |
|---|---|---|---|---|---|---|
| 1 | basic | 160 | 160 | 0.983 ± 0.027 | 97% | 0.983 |
| 1 | cosine | 160 | 160 | 0.948 ± 0.059 | 90% | 0.948 |
| 2 | basic | 80 | 160 | 0.957 ± 0.051 | 92% | 0.992 |
| 2 | cosine | 80 | 160 | 0.734 ± 0.169 | 48% | 0.828 |
| 4 | basic | 40 | 160 | 0.870 ± 0.034 | 71% | 0.980 |
| 4 | cosine | 40 | 160 | 0.584 ± 0.195 | 12% | 0.731 |
| 8 | basic | 20 | 160 | 0.521 ± 0.061 | 0% | 0.745 |
| 8 | cosine | 20 | 160 | 0.460 ± 0.184 | 1% | 0.687 |
| 16 | basic | 10 | 160 | 0.134 ± 0.031 | 0% | 0.279 |
| 16 | cosine | 10 | 160 | 0.329 ± 0.162 | 0% | 0.572 |
