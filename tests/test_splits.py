"""Tests for the image splits (src/splits.py)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import splits


def test_journal_splits_are_disjoint():
    assert splits.JOURNAL.is_disjoint(), splits.JOURNAL.overlaps()
    used = [set(s.indices()) for s in splits.JOURNAL.spans().values()]
    assert sum(len(u) for u in used) == len(set().union(*used))
    assert all(len(u) > 0 for u in used)


def test_legacy_splits_match_the_old_hard_coded_ranges():
    assert splits.LEGACY.is_disjoint()
    assert splits.LEGACY.prior.start == 82  # old POOL_START
    assert splits.LEGACY.targets.contains(50, 32)  # K sweep: test 50-81
    assert not splits.LEGACY.targets.contains(50, 33)
    assert splits.LEGACY.tuning.end() == 50  # tuning: test 0-49
    # The old CIFAR-10 baseline table reused tuning images; that is why it is legacy.
    assert splits.LEGACY_CIFAR10_BASELINE.overlaps() == [("tuning", "targets"), ("tuning", "prior")]


def test_overlap_detection():
    a, b = splits.Span(0, 10), splits.Span(10, None)
    assert not a.overlaps(b) and a.overlaps(splits.Span(9, 11))
    assert splits.Span(5, None).overlaps(splits.Span(9_000, 9_001))
    assert str(a) == "0-9" and str(b) == "10-end"


def test_class_gallery_has_every_image_of_the_class():
    labels = np.array([3, 1, 3, 0, 3, 1])
    g = splits.class_galleries(labels, num_classes=4)
    assert g[3].tolist() == [0, 2, 4] and g[2].tolist() == []
    assert sorted(np.concatenate(list(g.values())).tolist()) == list(range(len(labels)))


def test_pool_by_class_stays_inside_the_span():
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    pools = splits.pool_by_class(labels, splits.Span(4, None), num_classes=2)
    assert pools[0].tolist() == [4, 6] and pools[1].tolist() == [5, 7]
