# Attack-vs-defense evaluation (n=25 cifar10 images x 3 seeds, DP σ=1.0, clip=1.0)

| Attacker | Defense | SSIM (mean±std) ↓ | PSNR ↓ | MSE ↑ | Attack success (SSIM≥0.9) ↓ | Label acc ↓ |
|---|---|---|---|---|---|---|
| basic | none | 0.615 ± 0.480 | 32.28 | 0.1133 | 61% | 100% |
| basic | dp | 0.003 ± 0.010 | 5.01 | 0.3176 | 0% | 12% |
| basic | mask | 0.005 ± 0.009 | 4.99 | 0.3189 | 0% | 33% |
| adaptive | none | 0.615 ± 0.480 | 32.28 | 0.1133 | 61% | 100% |
| adaptive | dp | 0.002 ± 0.009 | 4.98 | 0.3195 | 0% | 100% |
| adaptive | mask | 0.003 ± 0.010 | 4.97 | 0.3204 | 0% | 100% |
| cosine | none | 0.708 ± 0.294 | 21.03 | 0.0233 | 53% | 100% |
| cosine | dp | 0.005 ± 0.010 | 5.42 | 0.2898 | 0% | 100% |
| cosine | mask | 0.009 ± 0.010 | 5.49 | 0.2851 | 0% | 100% |
