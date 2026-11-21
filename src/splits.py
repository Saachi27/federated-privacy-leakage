"""Which test images each part of the study may use, in one place."""

from dataclasses import dataclass
from typing import Dict, Optional

import hashlib
import json
import os

import numpy as np

N_TEST = 10_000  # MNIST and CIFAR-10 test sets


@dataclass(frozen=True)
class Span:
    """Test images start, start + 1, ..., stop - 1 (stop=None: to the end)."""
    start: int
    stop: Optional[int]

    def end(self, n_test=N_TEST):
        return n_test if self.stop is None else self.stop

    def indices(self, n_test=N_TEST):
        return np.arange(self.start, self.end(n_test))

    def contains(self, first, num, n_test=N_TEST):
        """True if images first .."""
        return self.start <= first and first + num <= self.end(n_test)

    def overlaps(self, other, n_test=N_TEST):
        return max(self.start, other.start) < min(self.end(n_test), other.end(n_test))

    def __str__(self):
        return f"{self.start}-{'end' if self.stop is None else self.stop - 1}"


@dataclass(frozen=True)
class Splits:
    name: str
    tuning: Span
    targets: Span
    prior: Span

    def spans(self):
        return {"tuning": self.tuning, "targets": self.targets, "prior": self.prior}

    def overlaps(self, n_test=N_TEST):
        """Pairs of uses that share at least one image (empty if disjoint)."""
        items = list(self.spans().items())
        return [(a, b) for i, (a, sa) in enumerate(items) for b, sb in items[i + 1:]
                if sa.overlaps(sb, n_test)]

    def is_disjoint(self, n_test=N_TEST):
        return not self.overlaps(n_test)


# Conference version: keeps existing results reproducible.
LEGACY = Splits("legacy", tuning=Span(0, 50), targets=Span(50, 82), prior=Span(82, None))
# Scripts/baselines_cifar10.py scored CIFAR-10 test 0-24
LEGACY_CIFAR10_BASELINE = Splits("legacy-cifar10-baseline", tuning=Span(0, 50),
                                 targets=Span(0, 25), prior=Span(25, None))


# Seeds of the seeded order, fixed before anyone looked at the result.
JOURNAL_SEEDS = {"mnist": 20261121, "cifar10": 20261122}
JOURNAL_TUNING = Span(0, 50)  # tuning images 0-49
JOURNAL_RETIRED = Span(50, 82)  # old study targets 50-81: never used again
JOURNAL_ELIGIBLE = Span(82, None)  # 82-9,999: targets and the baseline prior
RELEASES_PER_N = 30
# Where each N's targets sit in the seeded order P.
TARGET_SLICES = {2: (0, 60), 4: (60, 180)}
N_TARGETS = 180
SPLIT_FILE = "results/journal/splits_{dataset}.json"
ROLES = ("tuning", "retired", "targets", "prior")


@dataclass(frozen=True)
class JournalSplits:
    """The four disjoint index lists, plus the release groups."""
    dataset: str
    seed: int
    tuning: tuple
    retired: tuple
    order: tuple

    @property
    def targets(self):
        return self.order[:N_TARGETS]

    @property
    def prior(self):
        return self.order[N_TARGETS:]

    def lists(self):
        return {"tuning": list(self.tuning), "retired": list(self.retired),
                "targets": list(self.targets), "prior": list(self.prior)}

    def targets_for(self, n):
        lo, hi = TARGET_SLICES[n]
        return list(self.order[lo:hi])

    def releases(self, n):
        """The 30 releases for N images, in order: release r is targets_for(n)[n*r : n*r + n]."""
        t = self.targets_for(n)
        assert len(t) == n * RELEASES_PER_N
        return [t[n * r:n * (r + 1)] for r in range(RELEASES_PER_N)]


def journal_splits(dataset="mnist", n_test=N_TEST):
    """Build the journal splits from the seed."""
    seed = JOURNAL_SEEDS[dataset]
    order = np.random.default_rng(seed).permutation(JOURNAL_ELIGIBLE.indices(n_test))
    s = JournalSplits(dataset, seed, tuple(int(i) for i in JOURNAL_TUNING.indices(n_test)),
                      tuple(int(i) for i in JOURNAL_RETIRED.indices(n_test)),
                      tuple(int(i) for i in order))
    assert_disjoint(s.lists(), n_test)
    return s


def assert_disjoint(lists, n_test=N_TEST):
    """The four roles are pairwise disjoint, have no repeats, and together cover the test set."""
    sets = {}
    for role in ROLES:
        values = [int(i) for i in lists[role]]
        assert len(values) == len(set(values)), f"{role} has a repeated index"
        assert all(0 <= i < n_test for i in values), f"{role} has an index outside 0-{n_test - 1}"
        sets[role] = set(values)
    for i, a in enumerate(ROLES):
        for b in ROLES[i + 1:]:
            shared = sets[a] & sets[b]
            assert not shared, f"{a} and {b} share {len(shared)} images, e.g. {sorted(shared)[:5]}"
    assert sum(len(v) for v in sets.values()) == n_test, "the four roles do not cover the test set"


def _split_file_bytes(s):
    data = {"dataset": s.dataset, "seed": s.seed,
            "rule": "P = default_rng(seed).permutation(arange(82, 10000)); "
                    "targets = P[0:180] (N=2: P[0:60], N=4: P[60:180]); prior = P[180:]",
            **s.lists()}
    return (json.dumps(data, indent=1) + "\n").encode()


def write_split_file(dataset="mnist", path=None):
    """Write the split file once, from the seed."""
    path = path or SPLIT_FILE.format(dataset=dataset)
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists; the split file is written once and never overwritten")
    blob = _split_file_bytes(journal_splits(dataset))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "xb") as f:
        f.write(blob)
    return path, hashlib.sha256(blob).hexdigest()


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_split_file(dataset="mnist", path=None, check_seed=True):
    """Read the split file, assert the four lists are disjoint, and return JournalSplits."""
    path = path or SPLIT_FILE.format(dataset=dataset)
    with open(path) as f:
        data = json.load(f)
    assert_disjoint(data)
    s = JournalSplits(data["dataset"], int(data["seed"]), tuple(data["tuning"]), tuple(data["retired"]),
                      tuple(data["targets"]) + tuple(data["prior"]))
    if check_seed:
        assert s == journal_splits(dataset), f"{path} does not match the seeded order"
    return s


JOURNAL = journal_splits("mnist")

SPLITS = {s.name: s for s in (LEGACY,)}


def get(name):
    return SPLITS[name]


def pool_by_class(test_labels, span, num_classes=10) -> Dict[int, np.ndarray]:
    """Indices in `span` (a Span or a list of indices) grouped by class, for baseline guesses."""
    labels = np.asarray(test_labels)
    idx = span.indices(len(labels)) if isinstance(span, Span) else np.asarray(span, dtype=int)
    return {c: idx[labels[idx] == c] for c in range(num_classes)}


def class_gallery(test_labels, c):
    """All test images of class `c` (the retrieval gallery; includes targets)."""
    return np.flatnonzero(np.asarray(test_labels) == c)


def class_galleries(test_labels, num_classes=10) -> Dict[int, np.ndarray]:
    return {c: class_gallery(test_labels, c) for c in range(num_classes)}


if __name__ == "__main__":
    # Python -m src.splits [mnist|cifar10]: write the split file once and print its SHA-256.
    import sys
    path, sha = write_split_file(sys.argv[1] if len(sys.argv) > 1 else "mnist")
    print(f"{path}  sha256 {sha}")
