# One-Release ε Under Replace-One Adjacency

Made by `scripts/recompute_epsilon.py` from the saved results; no attack was rerun and no result file was changed.

- **Mechanism:** clip each example's gradient to norm C, sum the K clipped gradients, add Gaussian noise N(0, (σC)²) once, divide by K (`scripts/attack_noise_sum.py`).
- **Old ε (add/remove one example):** sensitivity C, so the accountant gets noise multiplier σ. This is what the study JSON, `summary.md` and the earlier paper draft report.
- **Replace-one ε:** one example is swapped for another, so K is unchanged and can be public. The sum moves by up to 2C, so the noise is σ/2 times the sensitivity: the accountant gets σ/2.
- Same accountant as before: Opacus RDP, sampling rate 1, one step, δ = 1e-05. ε is an accounting upper bound for one release; it is not the DP-SGD training ε.
- Replace-one ε at σ equals add/remove ε at σ/2. All saved ε values were checked to match the old formula.

| σ | Old ε (add/remove, σ) | Replace-one ε (σ/2) | Ratio | Used in |
|---|---|---|---|---|
| 0 | ∞ | ∞ | — | study grid (all settings) and paper |
| 3e-05 | 6.11e+08 | 2.44e+09 | 4.00× | script default only (not in the study) |
| 0.0001 | 5.50e+07 | 2.20e+08 | 4.00× | study grid (all settings) and paper |
| 0.0003 | 6.11e+06 | 2.44e+07 | 4.00× | study grid (all settings) and paper |
| 0.001 | 5.50e+05 | 2.20e+06 | 4.00× | study grid (all settings) and paper |
| 0.003 | 6.12e+04 | 2.45e+05 | 3.99× | study grid (all settings) and paper |
| 0.01 | 5.61e+03 | 2.21e+04 | 3.94× | study grid (all settings) and paper |
| 0.03 | 722 | 2.56e+03 | 3.54× | study grid (all settings) and paper |
| 0.1 | 96.1 | 295 | 3.07× | study grid (all settings) and paper |
| 0.3 | 20.4 | 52.6 | 2.58× | reconstruction bound (analyze_rq4.py; σ = 1 is in the paper) |
| 1 | 4.73 | 10.7 | 2.27× | reconstruction bound (analyze_rq4.py; σ = 1 is in the paper) |
| 2 | 2.17 | 4.73 | 2.18× | reconstruction bound (analyze_rq4.py; σ = 1 is in the paper) |

## What changes in the reported numbers

- σ = 0.01 stops every single-step attack. Its one-release ε goes from 5.61e+03 to 2.21e+04.
- The bound's σ = 1 goes from ε = 4.73 to 10.7.
- For tiny σ, RDP ε grows like 1/σ², so halving σ multiplies ε by about 4. Near σ = 1 the ratio is smaller.
- Check: replace-one ε at σ = 2 (4.73) equals the old ε at σ = 1.
- The earlier paper draft already said the replace-one values would need twice the σ; this table gives those values.

## The illustrative reconstruction bound (MNIST)

ReRo bound for one release (`scripts/analyze_rq4.py`, Kaissis et al. 2023 closed form): success ≤ Φ(Φ⁻¹(κ) + Δ/σ), with κ̂ = 0.0034 (`results/study/kappa_mnist.json`). Add/remove uses Δ = 1 (what the earlier draft reports). Replace-one uses Δ = 2, which is the add/remove bound at σ/2.

| σ | Bound, add/remove (Δ = 1) | Bound, replace-one (Δ = 2) |
|---|---|---|
| 0.3 | 73.5% | 100.0% |
| 1 | 4.4% | 24.0% |
| 2 | 1.4% | 4.4% |

- Which value to report: the cited ReRo results (Hayes et al. 2023; Kaissis et al. 2023) compare the release that holds the target with a reference release that does not depend on the target (add-one). Any such reference works, so for a fixed-size release (public K) one can take the other K − 1 clipped gradients plus the same noise, with the same divisor K. The target then shifts the mean by at most C/K against noise σC/K, so Δ = 1 holds and the add/remove column is a valid bound. The replace-one column (two candidate targets) is a valid but looser bound. Report the add/remove value (4.4% at σ = 1) with this argument.
- This is separate from the ε adjacency above: the one-release ε of a fixed-size release is still stated under replace-one.
- κ̂ is a lower estimate from finitely many guesses, so both bounds are indicative, not certified.
