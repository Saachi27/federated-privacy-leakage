# Why N × 1 leaks more (post hoc)

*Made by `scripts/why_more.py` from the saved target releases. Descriptive only: not a locked endpoint, changes no outcome. No attack was rerun.*

g = the E = 1 release (mean gradient). Cosines are over all parameters. Every recomputed drift matched the saved `drift_cosine` within 1e-4.

| Setting | N | Releases | cos(N×1 E=5, g) median [min] | cos(1×N E=5, g) median [min] | Worst one-image client cos(δ_c, g_c) median | Spearman(drift, SSIM gap) |
|---|---|---|---|---|---|---|
| mnist-fresh | 2 | 30 | 1.000 [1.000] | 0.853 [0.390] | 1.000 | 0.17 |
| mnist-fresh | 4 | 30 | 1.000 [1.000] | 0.735 [-0.143] | 1.000 | 0.22 |
| mnist-trained | 2 | 30 | 0.999 [0.975] | 0.986 [-0.288] | 0.999 | -0.09 |
| mnist-trained | 4 | 30 | 1.000 [0.970] | 0.924 [-0.155] | 0.999 | 0.45 |

Reading: a one-image client's five local steps keep its update almost parallel to its own gradient (every client cos ≥ 0.99), so the N × 1 secure sum points where the E = 1 batch gradient points. For a direction-matching attacker that release is as exposed as E = 1, which matches its SSIM (MNIST fresh N = 2: 0.855 at E = 5 vs 0.859 at E = 1). One client training on all N images moves away from g, because the images interact through the shared weights.

Limits of this reading: across releases, how far 1 × N moves from g only weakly predicts the size of the SSIM gap (Spearman column; the sign is mixed). On MNIST trained at N = 2 the 1 × N release stays close to g (median 0.986) yet attacker 2 scores 0.355 on it, below attacker 1's 0.531, so part of the trained-model gap is attacker 2 finding the interacting 1 × N problem harder.
