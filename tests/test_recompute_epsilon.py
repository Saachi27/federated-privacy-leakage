"""Tests for the replace-one epsilon of scripts/recompute_epsilon.py."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.attack_noise_sum import epsilon_for
from scripts.recompute_epsilon import epsilon_replace_one
import scripts.recompute_epsilon as re


@pytest.mark.parametrize("sigma", [1e-3, 0.01, 0.1, 1.0, 2.0])
def test_replace_one_is_add_remove_at_half_sigma(sigma):
    assert epsilon_replace_one(sigma) == pytest.approx(epsilon_for(sigma / 2))


def test_replace_one_is_larger():
    for sigma in (1e-3, 0.1, 1.0):
        assert epsilon_replace_one(sigma) > epsilon_for(sigma)


def test_replace_one_at_two_sigma_equals_old_at_sigma():
    assert epsilon_replace_one(2.0) == pytest.approx(epsilon_for(1.0))


def test_no_noise_is_infinite():
    assert math.isinf(epsilon_replace_one(0.0))


def test_delta_is_passed_through():
    assert epsilon_replace_one(1.0, delta=1e-6) == pytest.approx(epsilon_for(0.5, delta=1e-6))
    assert epsilon_replace_one(1.0, delta=1e-6) > epsilon_replace_one(1.0, delta=1e-5)


def test_rero_replace_one_is_add_remove_at_half_sigma():
    from scripts.analyze_rq4 import rero_bound
    kappa = 0.0034
    assert re.rero_replace_one(1.0, kappa) == pytest.approx(rero_bound(0.5, kappa))
    assert re.rero_replace_one(1.0, kappa) > rero_bound(1.0, kappa)
    assert rero_bound(1.0, kappa) == pytest.approx(0.044, abs=5e-4)
    assert re.rero_replace_one(1.0, kappa) == pytest.approx(0.240, abs=5e-4)
    assert re.rero_replace_one(0.0, kappa) == 1.0
