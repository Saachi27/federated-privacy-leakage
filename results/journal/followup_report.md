# Follow-up checks

*Follow-up, written after the main result and the add-ons. Secondary: none of this changes the main outcome or any add-on label. Made by `scripts/analyze_followup.py` with the locked rules of `scripts/analyze_journal.py` (same outcome order, stratified paired t-intervals, ±0.05 margin). PC = the E = 1 positive control of the main run.*

## 22.A Attacker 2 with 8 restarts (E = 5)

Each block shows the locked 2-restart row and the 8-restart row. The 8 restarts include the main run's 2 starts.

| Setting | N | Attacker | Outcome | PC s − b (95%) | 1 × N s − b (95%) | N × 1 s − b (95%) | Gap N × 1 − 1 × N (90%) | Mean SSIM 1 × N / N × 1 | Any ≥ 0.9 1 × N / N × 1 | Retrieval 1 × N / N × 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| mnist-fresh | 2 | attacker 2 (2 restarts, locked) | **more** | 0.527 [0.458, 0.597] | 0.078 [-0.006, 0.163] | 0.524 [0.453, 0.594] | 0.445 [0.357, 0.534] | 0.410 / 0.855 | 0.10 / 0.90 | 0.93 / 1.00 |
| mnist-fresh | 2 | attacker 2, 8 restarts | **more** | 0.527 [0.458, 0.597] | 0.138 [0.056, 0.221] | 0.532 [0.463, 0.600] | 0.394 [0.314, 0.473] | 0.469 / 0.863 | 0.10 / 0.90 | 0.97 / 1.00 |
| mnist-trained | 2 | attacker 2 (2 restarts, locked) | **more** | 0.298 [0.250, 0.346] | 0.023 [-0.053, 0.099] | 0.366 [0.330, 0.402] | 0.343 [0.283, 0.402] | 0.355 / 0.697 | 0.10 / 0.63 | 0.73 / 1.00 |
| mnist-trained | 2 | attacker 2, 8 restarts | **more** | 0.298 [0.250, 0.346] | 0.096 [0.011, 0.181] | 0.399 [0.364, 0.433] | 0.303 [0.241, 0.365] | 0.427 / 0.730 | 0.20 / 0.80 | 0.83 / 1.00 |
| mnist-fresh | 4 | attacker 2 (2 restarts, locked) | **more** | 0.467 [0.406, 0.529] | -0.165 [-0.212, -0.118] | 0.465 [0.403, 0.527] | 0.630 [0.567, 0.692] | 0.148 / 0.778 | 0.00 / 0.80 | 0.47 / 1.00 |
| mnist-fresh | 4 | attacker 2, 8 restarts | **more** | 0.467 [0.406, 0.529] | -0.143 [-0.187, -0.098] | 0.477 [0.415, 0.539] | 0.620 [0.559, 0.681] | 0.170 / 0.790 | 0.00 / 0.77 | 0.57 / 1.00 |
| mnist-trained | 4 | attacker 2 (2 restarts, locked) | **more** | 0.126 [0.088, 0.164] | -0.031 [-0.087, 0.024] | 0.179 [0.137, 0.221] | 0.210 [0.154, 0.267] | 0.281 / 0.492 | 0.03 / 0.30 | 0.83 / 1.00 |
| mnist-trained | 4 | attacker 2, 8 restarts | **more** | 0.126 [0.088, 0.164] | 0.019 [-0.050, 0.087] | 0.214 [0.167, 0.261] | 0.195 [0.126, 0.264] | 0.332 / 0.527 | 0.07 / 0.43 | 0.87 / 1.00 |

Paired change in mean SSIM from 2 to 8 restarts (8 − 2; 90% and 95% stratified t-intervals; descriptive):

| Block | Split | Change (90%) | Change (95%) |
|---|---|---|---|
| mnist-fresh N=2 | 1 × 2 | 0.060 [0.018, 0.102] | 0.060 [0.009, 0.111] |
| mnist-fresh N=2 | 2 × 1 | 0.008 [-0.001, 0.017] | 0.008 [-0.003, 0.019] |
| mnist-trained N=2 | 1 × 2 | 0.072 [0.018, 0.127] | 0.072 [0.007, 0.138] |
| mnist-trained N=2 | 2 × 1 | 0.033 [0.014, 0.051] | 0.033 [0.010, 0.055] |
| mnist-fresh N=4 | 1 × 4 | 0.022 [0.008, 0.036] | 0.022 [0.006, 0.038] |
| mnist-fresh N=4 | 4 × 1 | 0.012 [0.003, 0.021] | 0.012 [0.001, 0.023] |
| mnist-trained N=4 | 1 × 4 | 0.050 [0.015, 0.085] | 0.050 [0.008, 0.092] |
| mnist-trained N=4 | 4 × 1 | 0.035 [0.024, 0.045] | 0.035 [0.022, 0.047] |

## 22.B Attacker 2 without the label-to-client map (N = 2, E = 5)

The N × 1 attacker tries both label orders and keeps the one with the lower final objective. The 1 × 2 side is the main run (it needs no map).

| Setting | N | Attacker | Outcome | PC s − b (95%) | 1 × N s − b (95%) | N × 1 s − b (95%) | Gap N × 1 − 1 × N (90%) | Mean SSIM 1 × N / N × 1 | Any ≥ 0.9 1 × N / N × 1 | Retrieval 1 × N / N × 1 | True order kept (different labels) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mnist-fresh | 2 | attacker 2, label map unknown | **more** | 0.527 [0.458, 0.597] | 0.078 [-0.006, 0.163] | 0.524 [0.453, 0.595] | 0.446 [0.357, 0.534] | 0.410 / 0.855 | 0.10 / 0.90 | 0.93 / 1.00 | 11/26 |
| mnist-trained | 2 | attacker 2, label map unknown | **more** | 0.298 [0.250, 0.346] | 0.023 [-0.053, 0.099] | 0.372 [0.335, 0.409] | 0.349 [0.290, 0.408] | 0.355 / 0.703 | 0.10 / 0.67 | 0.73 / 1.00 | 15/26 |
