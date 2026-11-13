# DP Accuracy Cost (RQ3)

MNIST, 10 clients, 20 rounds, batch 64, lr 0.01 unless stated. Final test accuracy, mean ± std over seeds. DP runs report ε per client at δ = 1e-5.

| Setting | Accuracy | n | ε |
|---|---|---|---|
| Plain FedAvg (no DP) | 0.9901 ± 0.0004 | 3 | — |
| Opacus on, no clip, no noise | 0.9896 ± 0.0003 | 3 | ∞ |
| Clip 1, no noise | 0.9126 ± 0.0036 | 3 | ∞ |
| σ=1, clip 1 (main DP run) | 0.9087 ± 0.0032 | 3 | 2.97 |
| σ=1, clip 2 | 0.9215 ± 0.0053 | 3 | 2.97 |
| σ=1, clip 5 | 0.9217 ± 0.0094 | 3 | 2.97 |
| σ=1, clip 1, lr 0.03 | 0.9227 ± 0.0078 | 3 | 2.97 |
| σ=1, clip 1, lr 0.1 | 0.9269 ± 0.0074 | 3 | 2.97 |

- **Clipping causes almost all of the loss.** Clipping to norm 1 with no noise drops accuracy from 0.9901 to 0.9126 (7.7 points). Adding σ = 1 noise costs only 0.4 more points. Opacus itself, with no clip and no noise, costs nothing (0.9896).
- **Tuning wins back about 1–2 points at the same ε.** A larger clip (2 or 5) or a larger learning rate (0.03 or 0.1) reaches 0.922–0.927. ε depends only on σ, sampling rate and steps, so these arms keep the ε of σ = 1.

Figures: `results/fig_dp_cost.png`, `results/fig_accuracy_vs_eps.png`.
