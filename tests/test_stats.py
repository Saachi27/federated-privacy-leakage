"""Tests for src/stats.py: stratified paired intervals, TOST and the outcome rules."""

import os
import sys

import numpy as np
import pytest
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import stats


def _data(seed=0, n_per=10, strata=3, shift=0.1):
    rng = np.random.default_rng(seed)
    s = np.repeat(np.arange(strata), n_per)
    b = rng.uniform(0.2, 0.4, len(s)) + 0.2 * s  # checkpoints differ a lot
    a = b + shift + rng.normal(0, 0.05, len(s))
    return a, b, s


def test_one_stratum_t_matches_scipy_paired_t():
    a, b, _ = _data(strata=1, n_per=12)
    ci = stats.paired_diff_interval(a, b, None, level=0.95, method="t")
    d = a - b
    lo, hi = sps.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=sps.sem(d))
    assert ci["mean"] == pytest.approx(d.mean())
    assert ci["lo"] == pytest.approx(lo)
    assert ci["hi"] == pytest.approx(hi)


def test_stratified_mean_is_size_weighted():
    a = np.array([1.0, 1.0, 1.0, 4.0])
    b = np.zeros(4)
    s = np.array([0, 0, 0, 1])
    ci = stats.paired_diff_interval(a, b, s, method="t")
    assert ci["mean"] == pytest.approx(7 / 4)
    assert ci["strata"] == 2 and ci["n"] == 4


def test_pairing_removes_checkpoint_offsets():
    a, b, s = _data(shift=0.1)
    for method in ("t", "bootstrap"):
        ci = stats.paired_diff_interval(a, b, s, method=method, n_resamples=2000)
        assert ci["lo"] < 0.1 < ci["hi"]
        assert ci["hi"] - ci["lo"] < 0.06


def test_bootstrap_is_reproducible_and_close_to_t():
    a, b, s = _data(n_per=30)
    one = stats.paired_diff_interval(a, b, s, method="bootstrap", n_resamples=4000, seed=7)
    two = stats.paired_diff_interval(a, b, s, method="bootstrap", n_resamples=4000, seed=7)
    t = stats.paired_diff_interval(a, b, s, method="t")
    assert one == two
    assert one["lo"] == pytest.approx(t["lo"], abs=0.01)
    assert one["hi"] == pytest.approx(t["hi"], abs=0.01)


def test_wider_level_gives_wider_interval():
    a, b, s = _data()
    n90 = stats.paired_diff_interval(a, b, s, level=0.90)
    n95 = stats.paired_diff_interval(a, b, s, level=0.95)
    assert n95["lo"] < n90["lo"] and n90["hi"] < n95["hi"]


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        stats.paired_diff_interval([1, 2], [1], None)
    with pytest.raises(ValueError):
        stats.paired_diff_interval([1, 2], [1, 2], [0])
    with pytest.raises(ValueError):
        stats.paired_diff_interval([1, 2], [1, 2], None, method="nope")


def test_tost_equivalent_and_not():
    a, b, s = _data(shift=0.0, n_per=40)
    eq = stats.tost(a, b, s, margin=0.05)
    assert eq["level"] == 0.90 and eq["equivalent"]
    a2, b2, s2 = _data(shift=0.05, n_per=40)
    assert not stats.tost(a2, b2, s2, margin=0.05)["equivalent"]  # interval crosses +0.05
    assert stats.tost(a2, b2, s2, margin=0.05, method="bootstrap", n_resamples=2000)["margin"] == 0.05


def test_beats_baseline():
    assert stats.beats_baseline(0.01)
    assert not stats.beats_baseline(0.0)
    assert not stats.beats_baseline(-0.2)
    assert not stats.beats_baseline(float("nan"))


# Classify_outcome(control_lo95, single_lo95, multi_lo95, gap_lo90, gap_hi90)
def test_outcome_not_tested_comes_first():
    # Everything else would say "same", but the E = 1 control fails.
    assert stats.classify_outcome(-0.01, 0.2, 0.2, -0.01, 0.01) == stats.NOT_TESTED


def test_outcome_neither_above_baseline():
    # Gap inside the margin, but neither split beats the baseline: not "same".
    assert stats.classify_outcome(0.3, -0.1, -0.05, -0.01, 0.01) == stats.NEITHER


def test_outcome_less():
    assert stats.classify_outcome(0.3, 0.2, -0.1, -0.3, -0.1) == stats.LESS


def test_outcome_less_needs_single_client_above_baseline():
    # Interval below -margin, but only the multi-client split beats the baseline.
    assert stats.classify_outcome(0.3, -0.1, 0.1, -0.3, -0.1) == stats.INCONCLUSIVE


def test_outcome_same():
    assert stats.classify_outcome(0.3, 0.2, 0.15, -0.04, 0.03) == stats.SAME


def test_outcome_same_needs_both_splits_above_baseline():
    assert stats.classify_outcome(0.3, 0.2, -0.01, -0.04, 0.03) == stats.INCONCLUSIVE


def test_outcome_inconclusive_when_interval_crosses_margin():
    assert stats.classify_outcome(0.3, 0.2, 0.15, -0.08, 0.02) == stats.INCONCLUSIVE


def test_outcome_margin_is_strict():
    assert stats.classify_outcome(0.3, 0.2, 0.15, -0.05, 0.03) == stats.INCONCLUSIVE
    assert stats.classify_outcome(0.3, 0.2, 0.15, -0.2, -0.05) == stats.INCONCLUSIVE


def test_outcome_more():
    assert stats.classify_outcome(0.3, 0.1, 0.4, 0.1, 0.3) == stats.MORE


def test_outcome_more_needs_multi_client_above_baseline():
    # Interval above +margin, but only the single-client split beats the baseline.
    assert stats.classify_outcome(0.3, 0.1, -0.01, 0.1, 0.3) == stats.INCONCLUSIVE


def test_outcome_more_margin_is_strict():
    assert stats.classify_outcome(0.3, 0.1, 0.4, 0.05, 0.3) == stats.INCONCLUSIVE


def test_outcome_more_comes_after_neither_and_not_tested():
    assert stats.classify_outcome(-0.1, 0.1, 0.4, 0.1, 0.3) == stats.NOT_TESTED
    assert stats.classify_outcome(0.3, -0.1, -0.1, 0.1, 0.3) == stats.NEITHER


def test_outcome_order_is_the_locked_order():
    assert stats.OUTCOME_ORDER == ["not tested", "neither above baseline", "less", "more", "same",
                                   "inconclusive"]


def test_too_few_releases_per_checkpoint_gives_no_label():
    s = np.repeat([0, 1, 2], 10)
    assert stats.enough_releases(s) and stats.releases_per_checkpoint(s) == {0: 10, 1: 10, 2: 10}
    assert stats.enough_releases(np.repeat([0, 1, 2], 5))
    assert not stats.enough_releases(np.array([0] * 10 + [1] * 10 + [2] * 4))
    assert not stats.enough_releases(np.repeat([0, 1], 10), checkpoints=[0, 1, 2])  # checkpoint 2 missing
    assert not stats.enough_releases(np.array([], int))
    assert stats.classify_outcome(0.3, 0.2, 0.15, -0.04, 0.03, enough=False) == stats.DESCRIPTIVE
