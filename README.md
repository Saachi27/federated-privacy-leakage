# One-Image Clients Leak More Under Secure Aggregation

Code, saved attack records and analysis scripts for the paper

> **One-Image Clients Leak More Under Secure Aggregation: A Paired Gradient-Inversion
> Measurement After Local Training**
> Saachi Badal, Riya Bachheti, Sneha, Greeshma Arya.
> Indira Gandhi Delhi Technical University for Women, Delhi, India.

## Overview

Secure aggregation hides individual client updates but releases their sum. We
compare two allocations of the same images, on the same model checkpoints, after
E = 5 local SGD steps per client:

- **N × 1**: N clients with one image each; the server sees only the sum.
- **1 × N**: one client with all N images (a single-client reference).

Two gradient-inversion attackers are used: gradient matching against the released
aggregate (attacker 1) and differentiable unrolling of the local steps (attacker 2).
Each comparison includes an E = 1 positive control and a label-aware per-release
baseline. The decision rules were fixed before the target runs.

A second study (Study A) measures one-step aggregation over K examples, Gaussian
noise on the aggregate, and the accuracy cost of DP-SGD.

## Main result

Fresh MNIST, N = 2, E = 5, attacker 2, 30 paired releases on three checkpoints:

| Split | Mean SSIM | Releases with an image at SSIM ≥ 0.9 |
|---|---|---|
| 2 × 1 (two clients) | 0.855 | 27 / 30 |
| 1 × 2 (one client) | 0.410 | 3 / 30 |

The paired gap is 0.445 (90% interval [0.357, 0.534]), and the outcome is "more"
under the prespecified rule. Attacker 2 gives the same outcome in all four MNIST
cells (fresh and trained models, N = 2 and 4), with eight restarts, and at N = 2
without the label-to-client map. A post hoc best-of-both-attackers check narrows
the trained-model gaps.

Full tables:

- [`results/journal/report.md`](results/journal/report.md): main analysis
- [`results/journal/followup_report.md`](results/journal/followup_report.md): eight restarts and unknown label map
- [`results/journal/addons_report.md`](results/journal/addons_report.md): 2 × 2 split, E = 10 and CIFAR-10
- [`results/journal/best_of_both.md`](results/journal/best_of_both.md): best-of-both-attackers check
- [`results/study/summary.md`](results/study/summary.md): Study A

## Repository layout

| Path | Contents |
|---|---|
| `src/` | FedAvg clients and server, models, attacks, multi-client releases, masking, DP, metrics (SSIM, PSNR) and statistics |
| `scripts/` | Experiment runners, analysis and figure scripts; `run_experiments.sh` runs each stage |
| `configs/` | Training configurations for MNIST and CIFAR-10 |
| `tests/` | Unit tests |
| `results/journal/` | Client-split study (Study B): attack records, reports and figures |
| `results/study/` | Study A: per-setting results and summary |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The results were produced with Python 3.9, PyTorch 2.8.0, torchvision 0.23.0,
Opacus 1.6.0, NumPy 2.0.2 and SciPy 1.13.1 on CPU. MNIST and CIFAR-10 are
downloaded by torchvision into `data/`.

## Reproducing the results

Regenerate the analyses and figures from the saved records (no attack is rerun;
about a minute):

```bash
bash scripts/run_experiments.sh journal
python scripts/analyze_followup.py
python scripts/analyze_addons.py && python scripts/make_addon_figures.py
python scripts/analyze_best_of_both.py
```

Rerun the attacks (CPU, several hours):

```bash
bash scripts/run_experiments.sh pilot        # tuning images: learning-rate rule and TV weights
bash scripts/run_experiments.sh multiclient  # 960 target tasks, about 4.8 h on an 8-core laptop
bash scripts/run_experiments.sh followup     # eight restarts and unknown label map
bash scripts/run_experiments.sh addons       # 2 x 2, E = 10 and CIFAR-10
bash scripts/run_experiments.sh study        # Study A
```

Finished tasks are skipped, so a run can be stopped and resumed. The three
trained MNIST checkpoints used by the attacks are in `results/models/`.

## Tests

```bash
python -m pytest -q
```

## License

MIT. See [`LICENSE`](LICENSE).
