# Attack-vs-defense evaluation (n=3 mnist images x 3 seeds)

| Defense | SSIM (mean±std) ↓ | PSNR ↓ | MSE ↑ | Attack success (SSIM≥0.9) ↓ | Label acc ↓ |
|---|---|---|---|---|---|
| none | 0.999 ± 0.001 | 71.35 | 0.0000 | 100% | 100% |
| dp | 0.003 ± 0.018 | 3.13 | 0.4866 | 0% | 11% |
| mask | 0.014 ± 0.013 | 3.23 | 0.4758 | 0% | 11% |
