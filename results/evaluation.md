# Attack-vs-defense evaluation (n=50 mnist images x 3 seeds, DP σ=1.0, clip=1.0)

| Attacker | Defense | SSIM (mean±std) ↓ | PSNR ↓ | MSE ↑ | Attack success (SSIM≥0.9) ↓ | Label acc ↓ |
|---|---|---|---|---|---|---|
| basic | none | 0.990 ± 0.056 | 73.40 | 0.0000 | 97% | 100% |
| basic | dp | 0.002 ± 0.013 | 3.12 | 0.4875 | 0% | 6% |
| basic | mask | 0.002 ± 0.014 | 3.18 | 0.4809 | 0% | 24% |
| adaptive | none | 0.990 ± 0.056 | 73.40 | 0.0000 | 97% | 100% |
| adaptive | dp | 0.003 ± 0.013 | 3.16 | 0.4832 | 0% | 100% |
| adaptive | mask | 0.004 ± 0.016 | 3.21 | 0.4776 | 0% | 100% |
| cosine | none | 0.922 ± 0.143 | 46.49 | 0.0006 | 77% | 100% |
| cosine | dp | 0.003 ± 0.015 | 3.61 | 0.4355 | 0% | 100% |
| cosine | mask | 0.010 ± 0.015 | 3.83 | 0.4149 | 0% | 100% |
