# Pilot timing and projection

*Made by `scripts/pilot_timing.py` from the pilot records (tuning images only). 4 workers, 1 torch thread each, 2 restarts, fixed steps.*

## Measured: mean worker-seconds per task

| Setting | N | Cell | Tasks timed | Mean s | Min s | Max s |
|---|---|---|---|---|---|---|
| mnist-fresh | 2 | E1_1x2_cosine | 25 | 21 | 19 | 22 |
| mnist-fresh | 2 | E1_1x2_lbfgs | 25 | 13 | 7 | 43 |
| mnist-fresh | 2 | E5_1x2_cosine | 25 | 21 | 18 | 23 |
| mnist-fresh | 2 | E5_1x2_lbfgs | 25 | 38 | 12 | 51 |
| mnist-fresh | 2 | E5_1x2_unrolled | 25 | 96 | 87 | 100 |
| mnist-fresh | 2 | E5_2x1_cosine | 25 | 21 | 19 | 23 |
| mnist-fresh | 2 | E5_2x1_lbfgs | 25 | 39 | 3 | 53 |
| mnist-fresh | 2 | E5_2x1_unrolled | 25 | 122 | 99 | 127 |
| mnist-fresh | 4 | E1_1x4_cosine | 4 | 34 | 31 | 35 |
| mnist-fresh | 4 | E1_1x4_lbfgs | 4 | 33 | 22 | 44 |
| mnist-fresh | 4 | E5_1x4_cosine | 4 | 33 | 31 | 35 |
| mnist-fresh | 4 | E5_1x4_lbfgs | 4 | 83 | 79 | 86 |
| mnist-fresh | 4 | E5_1x4_unrolled | 4 | 158 | 155 | 166 |
| mnist-fresh | 4 | E5_4x1_cosine | 4 | 35 | 33 | 37 |
| mnist-fresh | 4 | E5_4x1_lbfgs | 4 | 74 | 41 | 87 |
| mnist-fresh | 4 | E5_4x1_unrolled | 4 | 226 | 176 | 251 |
| mnist-trained | 2 | E1_1x2_cosine | 2 | 19 | 19 | 20 |
| mnist-trained | 2 | E1_1x2_lbfgs | 2 | 7 | 3 | 11 |
| mnist-trained | 2 | E5_1x2_cosine | 2 | 20 | 19 | 20 |
| mnist-trained | 2 | E5_1x2_lbfgs | 2 | 14 | 8 | 20 |
| mnist-trained | 2 | E5_1x2_unrolled | 2 | 89 | 89 | 90 |
| mnist-trained | 2 | E5_2x1_cosine | 2 | 20 | 20 | 20 |
| mnist-trained | 2 | E5_2x1_lbfgs | 2 | 19 | 10 | 28 |
| mnist-trained | 2 | E5_2x1_unrolled | 2 | 105 | 94 | 116 |

Pilot N = 2 wall time: 40.4 min for 200 tasks (packing 0.96: share of 4 workers kept busy).

## One release, all cells (worker-seconds)

| Setting | N | Per release | 30 releases (worker-hours) | Source |
|---|---|---|---|---|
| mnist-fresh | 2 | 371 s | 3.1 | measured (mnist-fresh, N = 2) |
| mnist-trained | 2 | 294 s | 2.4 | measured (mnist-trained, N = 2) |
| mnist-fresh | 4 | 676 s | 5.6 | measured (mnist-fresh, N = 4) |
| mnist-trained | 4 | 676 s | 5.6 | = MNIST fresh, N = 4 |

## Full target grid at each cut level (10-hour nights, 5 nights = 50 h)

| Cut level | Blocks kept | Worker-hours | Wall hours | Nights | Fits in 5 nights? |
|---|---|---|---|---|---|
| none | fresh N=2, trained N=2, fresh N=4, trained N=4 | 16.8 | 4.4 | 0.44 | yes |
| 1 (drop trained N=4) | fresh N=2, trained N=2, fresh N=4 | 11.2 | 2.9 | 0.29 | yes |
| 2 (also fresh N=4) | fresh N=2, trained N=2 | 5.5 | 1.4 | 0.14 | yes |
| 3 (also trained N=2) | fresh N=2 | 3.1 | 0.8 | 0.08 | yes |

## Stop and resume (plan row F)

- Run: `scripts/attack_multiclient.py --releases pilot --settings mnist-fresh --ns 2 --workers 4` (25 tuning pairs, all 5 cells, both attacker-1 objectives, 2 restarts: 200 tasks).
- Segment 1: 714 s, stopped on purpose at 62/200 tasks done (SIGTERM to the runner and its workers; note that a `nohup ... &` job ignores SIGINT, so Ctrl-C style stops must be done in the foreground or with SIGTERM).
- After the stop: no worker left, no partial `.tmp` file, 62 records with their npz.
- Segment 2 (same command): "200 tasks, 62 already done, 138 to run"; all 62 finished records passed the settings check; 138 tasks in 1,710 s; 0 failures. Resume works.
- Total wall 2,424 s (40.4 min) for 25 releases. Scaled by 30/25 for the minimum grid: 48.5 min.

## Other timings

- N = 4, MNIST fresh: 4 tuning groups of 4 (images 0–15), all cells: 32 tasks in 778 s wall.
- N = 2, MNIST trained: 2 tuning pairs, all cells: 16 tasks in 197 s wall. Trained is a little cheaper than fresh (its L-BFGS runs stop early). Trained N = 4 is estimated as equal to fresh N = 4.

## Caveats

- The laptop was not idle. During the TV tuning (02:37–07:21) other jobs (test runs in other sessions) pushed the load average to 10–12, and attacker-2 tasks took 140–400 s instead of about 100 s. The grid timings above were taken at a load average of about 3–5. Even a 3× slowdown keeps the full grid (cut level none) under 15 h, i.e. 2 nights.
- With 10-hour nights the full grid (no cuts) needs about 0.45 nights; with 8-hour nights about 0.55. So no cut is needed, and the pilot shows spare time.
