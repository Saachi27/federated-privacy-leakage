"""Tests for the checkpoint hashes (src/checkpoints.py)."""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import checkpoints
from src.settings import build_target

META = {"in_channels": 1, "num_classes": 10, "input_size": 28}


@pytest.mark.parametrize("k", [0, 1, 2])
def test_fresh_checkpoints_match_their_recorded_hash(k):
    model = build_target(META, k, None)
    assert checkpoints.verify(model, k) == checkpoints.FRESH_STATE_SHA256[k]


def test_a_changed_model_stops_the_run():
    model = build_target(META, 0, None)
    with torch.no_grad():
        next(model.parameters())[0].add_(1e-6)
    with pytest.raises(RuntimeError):
        checkpoints.verify(model, 0)
    with pytest.raises(RuntimeError):
        checkpoints.verify(build_target(META, 1, None), 0)  # the wrong seed


@pytest.mark.parametrize("k", [0, 1, 2])
def test_trained_checkpoints_match_the_plan(k):
    path = "results/models/mnist_lenet_r20_s{seed}.pt"
    if not os.path.exists(os.path.join(ROOT, path.format(seed=k))):
        pytest.skip("trained checkpoints not present")
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        model = build_target(META, k, path)
        assert checkpoints.verify(model, k, path) == checkpoints.TRAINED_SHA256[k]
    finally:
        os.chdir(cwd)


CIFAR_META = {"in_channels": 3, "num_classes": 10, "input_size": 32}


@pytest.mark.parametrize("k", [0, 1, 2])
def test_cifar10_fresh_checkpoints_match_their_recorded_hash(k):
    model = build_target(CIFAR_META, k, None)
    assert checkpoints.verify(model, k, dataset="cifar10") == checkpoints.CIFAR10_FRESH_STATE_SHA256[k]


def test_the_dataset_picks_the_table():
    with pytest.raises(RuntimeError):
        checkpoints.verify(build_target(CIFAR_META, 0, None), 0)  # CIFAR model vs MNIST table
    with pytest.raises(RuntimeError):
        checkpoints.verify(build_target(META, 0, None), 0, dataset="cifar10")
    with pytest.raises(RuntimeError):
        checkpoints.verify(build_target(CIFAR_META, 0, None), 0, "x_s{seed}.pt", dataset="cifar10")
