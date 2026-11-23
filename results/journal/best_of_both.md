# Best-of-both-attackers check (post hoc, secondary)

Per release, the higher mean SSIM of attacker 1 (cosine) and attacker 2, per split.
Gap = N x 1 - 1 x N, 90% checkpoint-stratified paired t-interval, 30 releases on 3 checkpoints.
Choosing the better attacker needs the true images: an evaluator's upper envelope, applied to both splits.
It does not change the locked outcome. Script: scripts/analyze_best_of_both.py.

| Setting | N | Attackers | Mean N x 1 | Mean 1 x N | Gap (90%) | Lower bound > +0.05 | Gap per checkpoint k0 / k1 / k2 |
|---|---|---|---|---|---|---|---|
| mnist-fresh | 2 | attacker 2 | 0.855 | 0.410 | 0.445 [0.357, 0.534] | yes | 0.637 / 0.463 / 0.236 |
| mnist-fresh | 2 | best of attacker 1 (cosine) and 2 | 0.858 | 0.415 | 0.443 [0.356, 0.530] | yes | 0.623 / 0.464 / 0.242 |
| mnist-fresh | 4 | attacker 2 | 0.778 | 0.148 | 0.630 [0.567, 0.692] | yes | 0.619 / 0.681 / 0.590 |
| mnist-fresh | 4 | best of attacker 1 (cosine) and 2 | 0.783 | 0.148 | 0.635 [0.573, 0.697] | yes | 0.625 / 0.685 / 0.595 |
| mnist-trained | 2 | attacker 2 | 0.697 | 0.355 | 0.343 [0.283, 0.402] | yes | 0.334 / 0.348 / 0.346 |
| mnist-trained | 2 | best of attacker 1 (cosine) and 2 | 0.702 | 0.573 | 0.129 [0.079, 0.178] | yes | 0.117 / 0.171 / 0.099 |
| mnist-trained | 4 | attacker 2 | 0.492 | 0.281 | 0.210 [0.154, 0.267] | yes | 0.168 / 0.260 / 0.204 |
| mnist-trained | 4 | best of attacker 1 (cosine) and 2 | 0.504 | 0.446 | 0.058 [0.019, 0.097] | no | 0.068 / 0.020 / 0.087 |
