# Attack on the sum over K examples (MNIST, fresh LeNet, 5 seeds)

| K | Attacker | Sums | Images | Mean SSIM (95% CI over seeds) | Success (SSIM≥0.9) | Best image per sum |
|---|---|---|---|---|---|---|
| 8 | basic | 20 | 160 | 0.536 ± 0.043 | 0% | 0.764 |
| 8 | cosine | 20 | 160 | 0.557 ± 0.142 | 4% | 0.795 |
| 16 | basic | 10 | 160 | 0.139 ± 0.034 | 0% | 0.286 |
| 16 | cosine | 10 | 160 | 0.437 ± 0.135 | 0% | 0.729 |
