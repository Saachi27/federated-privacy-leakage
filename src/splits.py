"""Which test images each part of the study may use, in one place."""

from dataclasses import dataclass
from typing import Dict, Optional

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

# Journal version.
JOURNAL_TUNING = Span(0, 100)
JOURNAL_TARGETS = Span(100, 1100)
JOURNAL_PRIOR = Span(1100, None)
JOURNAL = Splits("journal", tuning=JOURNAL_TUNING, targets=JOURNAL_TARGETS, prior=JOURNAL_PRIOR)

SPLITS = {s.name: s for s in (LEGACY, JOURNAL)}


def get(name):
    return SPLITS[name]


def pool_by_class(test_labels, span, num_classes=10) -> Dict[int, np.ndarray]:
    """Indices in `span` grouped by class (for baseline guesses from the prior)."""
    labels = np.asarray(test_labels)
    idx = span.indices(len(labels))
    return {c: idx[labels[idx] == c] for c in range(num_classes)}


def class_gallery(test_labels, c):
    """All test images of class `c` (the retrieval gallery; includes targets)."""
    return np.flatnonzero(np.asarray(test_labels) == c)


def class_galleries(test_labels, num_classes=10) -> Dict[int, np.ndarray]:
    return {c: class_gallery(test_labels, c) for c in range(num_classes)}
