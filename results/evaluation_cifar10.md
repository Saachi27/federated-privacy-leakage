# Attack-vs-defense evaluation (n=25 cifar10 images x 3 seeds, DP σ=1.0, clip=1.0)

| Attacker | Defense | SSIM (mean±std) ↓ | PSNR ↓ | MSE ↑ | Attack success (SSIM≥0.9) ↓ | Label acc ↓ |
|---|---|---|---|---|---|---|
| basic | none | 0.640 ± 0.447 | 32.28 | 0.1133 | 60% | 100% |
| basic | dp | 0.072 ± 0.021 | 5.01 | 0.3176 | 0% | 12% |
| basic | mask | 0.073 ± 0.021 | 4.99 | 0.3189 | 0% | 33% |
| adaptive | none | 0.640 ± 0.447 | 32.28 | 0.1133 | 60% | 100% |
| adaptive | dp | 0.071 ± 0.019 | 4.98 | 0.3195 | 0% | 100% |
| adaptive | mask | 0.072 ± 0.020 | 4.97 | 0.3204 | 0% | 100% |
| cosine | none | 0.558 ± 0.205 | 15.66 | 0.0396 | 0% | 100% |
| cosine | dp | 0.072 ± 0.018 | 5.41 | 0.2905 | 0% | 100% |
| cosine | mask | 0.075 ± 0.020 | 5.48 | 0.2857 | 0% | 100% |
