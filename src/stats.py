"""Release-level statistics for the client-split study."""

import numpy as np
from scipy import stats as sps

MARGIN = 0.05
NOT_TESTED = "not tested"
NEITHER = "neither above baseline"
LESS = "less"
MORE = "more"
SAME = "same"
INCONCLUSIVE = "inconclusive"
DESCRIPTIVE = "descriptive only"
OUTCOME_ORDER = [NOT_TESTED, NEITHER, LESS, MORE, SAME, INCONCLUSIVE]
MIN_PER_CHECKPOINT = 5


def _diffs_by_stratum(a, b, strata):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-D arrays of the same length (one value per release)")
    strata = np.asarray(strata) if strata is not None else np.zeros(len(a), int)
    if len(strata) != len(a):
        raise ValueError("strata must have one label per release")
    d = a - b
    return {s: d[strata == s] for s in sorted(set(strata.tolist()))}


def _weighted_mean(groups):
    n = sum(len(g) for g in groups.values())
    return float(sum(len(g) * g.mean() for g in groups.values()) / n)


def stratified_t_interval(groups, level):
    """T-interval for the size-weighted mean of the strata means."""
    n = sum(len(g) for g in groups.values())
    mean = _weighted_mean(groups)
    terms = []
    for g in groups.values():
        if len(g) >= 2:
            w = len(g) / n
            terms.append((w ** 2 * g.var(ddof=1) / len(g), len(g) - 1))
    if not terms:
        return mean, float("nan"), float("nan")
    var = sum(t for t, _ in terms)
    if var == 0:
        return mean, mean, mean
    df = var ** 2 / sum(t ** 2 / dof for t, dof in terms)
    half = float(sps.t.ppf(0.5 + level / 2, df) * np.sqrt(var))
    return mean, mean - half, mean + half


def stratified_bootstrap_interval(groups, level, n_resamples=10_000, seed=0):
    """Percentile interval: resample releases with replacement inside each stratum."""
    rng = np.random.default_rng(seed)
    n = sum(len(g) for g in groups.values())
    boot = np.zeros(n_resamples)
    for g in groups.values():
        idx = rng.integers(0, len(g), size=(n_resamples, len(g)))
        boot += len(g) / n * g[idx].mean(axis=1)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return _weighted_mean(groups), float(lo), float(hi)


def paired_diff_interval(a, b, strata=None, level=0.95, method="t", n_resamples=10_000, seed=0):
    """Interval for the mean release-level difference a - b, stratified by checkpoint."""
    groups = _diffs_by_stratum(a, b, strata)
    if method == "t":
        mean, lo, hi = stratified_t_interval(groups, level)
    elif method == "bootstrap":
        mean, lo, hi = stratified_bootstrap_interval(groups, level, n_resamples, seed)
    else:
        raise ValueError(f"unknown method {method!r}")
    return {"mean": mean, "lo": lo, "hi": hi, "level": level, "method": method,
            "n": int(sum(len(g) for g in groups.values())), "strata": len(groups)}


def tost(a, b, strata=None, margin=MARGIN, method="t", **kw):
    """Equivalence of a and b within +-margin, from the paired 90% interval of a - b."""
    ci = paired_diff_interval(a, b, strata, level=0.90, method=method, **kw)
    ci["margin"] = margin
    ci["equivalent"] = bool(-margin < ci["lo"] and ci["hi"] < margin)
    return ci


def beats_baseline(lo95):
    """The paired 95% interval of (attack - baseline) has its lower end above 0."""
    return bool(np.isfinite(lo95) and lo95 > 0)


def releases_per_checkpoint(strata):
    """{checkpoint: number of releases}."""
    values, counts = np.unique(np.asarray(strata), return_counts=True)
    return {v.item(): int(c) for v, c in zip(values, counts)}


def enough_releases(strata, minimum=MIN_PER_CHECKPOINT, checkpoints=None):
    """True if every checkpoint has at least `minimum` releases."""
    counts = releases_per_checkpoint(strata) if len(strata) else {}
    wanted = list(counts) if checkpoints is None else list(checkpoints)
    return bool(wanted) and all(counts.get(k, 0) >= minimum for k in wanted)


def classify_outcome(control_lo95, single_lo95, multi_lo95, gap_lo90, gap_hi90, margin=MARGIN,
                     enough=True):
    """Main-endpoint outcome, checked in the locked order."""
    if not enough:
        return DESCRIPTIVE
    single, multi = beats_baseline(single_lo95), beats_baseline(multi_lo95)
    if not beats_baseline(control_lo95):
        return NOT_TESTED
    if not single and not multi:
        return NEITHER
    if gap_hi90 < -margin and single:
        return LESS
    if gap_lo90 > margin and multi:
        return MORE
    if -margin < gap_lo90 and gap_hi90 < margin and single and multi:
        return SAME
    return INCONCLUSIVE
