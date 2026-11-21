"""Tests for the image splits (src/splits.py)."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import splits


def test_journal_splits_follow_plan_section_2():
    j = splits.JOURNAL
    order = np.random.default_rng(20261121).permutation(np.arange(82, 10000))
    assert list(j.tuning) == list(range(50)) and list(j.retired) == list(range(50, 82))
    assert list(j.order) == order.tolist()
    assert list(j.targets) == order[:180].tolist() and list(j.prior) == order[180:].tolist()
    assert j.targets_for(2) == order[0:60].tolist() and j.targets_for(4) == order[60:180].tolist()
    r2, r4 = j.releases(2), j.releases(4)
    assert len(r2) == len(r4) == 30
    assert all(r2[r] == [order[2 * r], order[2 * r + 1]] for r in range(30))
    assert all(r4[r] == order[60 + 4 * r:60 + 4 * r + 4].tolist() for r in range(30))
    # Releases never share an image, inside an N or across N.
    flat = [i for rel in r2 + r4 for i in rel]
    assert len(flat) == len(set(flat)) == 180


def test_journal_splits_are_disjoint_and_cover_the_test_set():
    lists = splits.JOURNAL.lists()
    splits.assert_disjoint(lists)
    assert [len(lists[k]) for k in splits.ROLES] == [50, 32, 180, 9738]
    bad = dict(lists, prior=lists["prior"] + [lists["targets"][0]])
    with pytest.raises(AssertionError):
        splits.assert_disjoint(bad)
    with pytest.raises(AssertionError):
        splits.assert_disjoint(dict(lists, prior=lists["prior"][1:]))  # one image left out


def test_split_file_is_written_once_and_reads_back(tmp_path):
    path = str(tmp_path / "splits_mnist.json")
    written, sha = splits.write_split_file("mnist", path)
    assert written == path and sha == splits.file_sha256(path)
    with pytest.raises(FileExistsError):
        splits.write_split_file("mnist", path)
    assert splits.load_split_file("mnist", path) == splits.JOURNAL


def test_committed_split_file_matches_the_seed():
    path = os.path.join(ROOT, "results", "journal", "splits_mnist.json")
    if not os.path.exists(path):
        pytest.skip("split file not written yet")
    assert splits.load_split_file("mnist", path) == splits.JOURNAL


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


def test_pool_by_class_takes_an_index_list():
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    pools = splits.pool_by_class(labels, [7, 2, 4], num_classes=2)
    assert pools[0].tolist() == [2, 4] and pools[1].tolist() == [7]
