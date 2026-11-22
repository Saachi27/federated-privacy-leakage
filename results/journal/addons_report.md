# Add-on experiments

*Add-on, written after the main result. Secondary: none of this changes the main outcome. Made by `scripts/analyze_addons.py` with the locked rules of `scripts/analyze_journal.py` (same outcome order, stratified paired t-intervals, ±0.05 margin).*

In each row the split with fewer images per client plays the "N × 1" role. PC = the E = 1 positive control of that setting and N.

| Group | Comparison | Setting | Attacker | Outcome | PC s − b (95%) | More-per-client split s − b (95%) | Fewer-per-client split s − b (95%) | Gap (90%) | Any image ≥ 0.9 (more / fewer) | Retrieval hit (more / fewer) |
|---|---|---|---|---|---|---|---|---|---|---|
| 21.A | G_a = 2×2 − 1×4 | mnist-fresh | attacker 2, cosine | **neither above baseline** | 0.467 [0.406, 0.529] | -0.165 [-0.212, -0.118] | -0.081 [-0.133, -0.030] | 0.084 [0.033, 0.134] | 0.00 / 0.00 | 0.47 / 0.83 |
| 21.A | G_a = 2×2 − 1×4 | mnist-fresh | attacker 1, cosine | **neither above baseline** | 0.467 [0.406, 0.529] | -0.269 [-0.293, -0.246] | -0.223 [-0.274, -0.172] | 0.046 [0.015, 0.077] | 0.00 / 0.00 | 0.07 / 0.23 |
| 21.A | G_a = 2×2 − 1×4 | mnist-fresh | attacker 1, basic | **neither above baseline** | 0.467 [0.406, 0.529] | -0.279 [-0.303, -0.255] | -0.276 [-0.301, -0.251] | 0.003 [-0.003, 0.010] | 0.00 / 0.00 | 0.00 / 0.03 |
| 21.A | G_b = 4×1 − 2×2 | mnist-fresh | attacker 2, cosine | **more** | 0.467 [0.406, 0.529] | -0.081 [-0.133, -0.030] | 0.465 [0.403, 0.527] | 0.546 [0.493, 0.599] | 0.00 / 0.80 | 0.83 / 1.00 |
| 21.A | G_b = 4×1 − 2×2 | mnist-fresh | attacker 1, cosine | **more** | 0.467 [0.406, 0.529] | -0.223 [-0.274, -0.172] | 0.462 [0.398, 0.525] | 0.685 [0.627, 0.743] | 0.00 / 0.80 | 0.23 / 1.00 |
| 21.A | G_b = 4×1 − 2×2 | mnist-fresh | attacker 1, basic | **neither above baseline** | 0.467 [0.406, 0.529] | -0.276 [-0.301, -0.251] | -0.264 [-0.291, -0.237] | 0.012 [0.004, 0.021] | 0.00 / 0.00 | 0.03 / 0.03 |
| 21.A | G_a = 2×2 − 1×4 | mnist-trained | attacker 2, cosine | **neither above baseline** | 0.126 [0.088, 0.164] | -0.031 [-0.087, 0.024] | -0.055 [-0.099, -0.011] | -0.024 [-0.072, 0.024] | 0.03 / 0.00 | 0.83 / 0.90 |
| 21.A | G_a = 2×2 − 1×4 | mnist-trained | attacker 1, cosine | **neither above baseline** | 0.126 [0.088, 0.164] | 0.038 [-0.029, 0.106] | 0.048 [-0.023, 0.120] | 0.010 [-0.036, 0.056] | 0.10 / 0.13 | 0.70 / 0.83 |
| 21.A | G_a = 2×2 − 1×4 | mnist-trained | attacker 1, basic | **neither above baseline** | 0.126 [0.088, 0.164] | -0.118 [-0.197, -0.040] | -0.119 [-0.192, -0.046] | -0.001 [-0.067, 0.065] | 0.00 / 0.03 | 0.40 / 0.53 |
| 21.A | G_b = 4×1 − 2×2 | mnist-trained | attacker 2, cosine | **more** | 0.126 [0.088, 0.164] | -0.055 [-0.099, -0.011] | 0.179 [0.137, 0.221] | 0.234 [0.203, 0.266] | 0.00 / 0.30 | 0.90 / 1.00 |
| 21.A | G_b = 4×1 − 2×2 | mnist-trained | attacker 1, cosine | **more** | 0.126 [0.088, 0.164] | 0.048 [-0.023, 0.120] | 0.166 [0.122, 0.210] | 0.117 [0.068, 0.167] | 0.13 / 0.37 | 0.83 / 1.00 |
| 21.A | G_b = 4×1 − 2×2 | mnist-trained | attacker 1, basic | **neither above baseline** | 0.126 [0.088, 0.164] | -0.119 [-0.192, -0.046] | -0.102 [-0.173, -0.032] | 0.017 [-0.042, 0.077] | 0.03 / 0.03 | 0.53 / 0.53 |
| 21.B | E = 10: 2×1 − 1×2 | mnist-fresh | attacker 2, cosine | **more** | 0.527 [0.458, 0.597] | -0.045 [-0.148, 0.057] | 0.523 [0.452, 0.594] | 0.569 [0.469, 0.669] | 0.10 / 0.90 | 0.63 / 1.00 |
| 21.B | E = 10: 2×1 − 1×2 | mnist-fresh | attacker 1, cosine | **more** | 0.527 [0.458, 0.597] | -0.163 [-0.266, -0.061] | 0.524 [0.453, 0.595] | 0.687 [0.592, 0.782] | 0.10 / 0.87 | 0.17 / 1.00 |
| 21.B | E = 10: 2×1 − 1×2 | mnist-fresh | attacker 1, basic | **neither above baseline** | 0.527 [0.458, 0.597] | -0.298 [-0.329, -0.267] | -0.294 [-0.328, -0.261] | 0.003 [-0.004, 0.011] | 0.00 / 0.00 | 0.03 / 0.03 |
| 21.C | CIFAR-10: 2×1 − 1×2 | cifar10-fresh | attacker 2, cosine | **more** | 0.367 [0.320, 0.414] | -0.004 [-0.080, 0.073] | 0.366 [0.319, 0.414] | 0.370 [0.295, 0.446] | 0.00 / 0.13 | 0.80 / 1.00 |
| 21.C | CIFAR-10: 2×1 − 1×2 | cifar10-fresh | attacker 1, cosine | **more** | 0.367 [0.320, 0.414] | 0.009 [-0.059, 0.077] | 0.381 [0.336, 0.426] | 0.372 [0.308, 0.437] | 0.00 / 0.00 | 0.73 / 1.00 |
| 21.C | CIFAR-10: 2×1 − 1×2 | cifar10-fresh | attacker 1, basic | **neither above baseline** | 0.367 [0.320, 0.414] | -0.138 [-0.164, -0.113] | -0.133 [-0.159, -0.106] | 0.006 [0.001, 0.010] | 0.00 / 0.00 | 0.03 / 0.13 |

Mean SSIM per split (the s in s − b) is the estimate's mean plus the baseline mean in the JSON.

