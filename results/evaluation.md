# Attack-vs-defense evaluation (n=50 mnist images x 3 seeds, DP σ=1.0, clip=1.0)

| Attacker | Defense | SSIM (mean±std) ↓ | PSNR ↓ | MSE ↑ | Attack success (SSIM≥0.9) ↓ | Label acc ↓ |
|---|---|---|---|---|---|---|
| basic | none | 0.998 ± 0.016 | 73.40 | 0.0000 | 99% | 100% |
| basic | dp | 0.002 ± 0.027 | 3.12 | 0.4875 | 0% | 6% |
| basic | mask | 0.002 ± 0.030 | 3.18 | 0.4809 | 0% | 24% |
| adaptive | none | 0.998 ± 0.016 | 73.40 | 0.0000 | 99% | 100% |
| adaptive | dp | 0.005 ± 0.030 | 3.16 | 0.4832 | 0% | 100% |
| adaptive | mask | 0.007 ± 0.036 | 3.21 | 0.4776 | 0% | 100% |
| cosine | none | 0.989 ± 0.017 | 37.37 | 0.0003 | 99% | 100% |
| cosine | dp | 0.006 ± 0.032 | 3.62 | 0.4345 | 0% | 100% |
| cosine | mask | 0.022 ± 0.033 | 3.87 | 0.4107 | 0% | 100% |
