"""Tests for the FedAvg core."""

import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.client import Client
from src.engine import run_federated
from src.models import SmallCNN
from src.server import average_states, select_clients


def test_average_states_is_weighted_mean():
    """Aggregation must equal the sample-weighted mean of weights."""
    a = {"w": torch.zeros(3)}
    b = {"w": torch.ones(3)}
    # 3:1 weighting -> mean = 0.25
    out = average_states([a, b], [3, 1])
    assert torch.allclose(out["w"], torch.full((3,), 0.25)), out["w"]


def test_average_states_identity():
    """Averaging identical states returns the same state."""
    s = {"w": torch.tensor([1.0, 2.0, 3.0])}
    out = average_states([s, s, s], [1, 1, 1])
    assert torch.allclose(out["w"], s["w"])


def test_select_clients_respects_fraction():
    import random
    rng = random.Random(0)
    sel = select_clients(10, 0.3, rng)
    assert len(sel) == 3
    assert len(set(sel)) == 3 and all(0 <= c < 10 for c in sel)


def test_client_local_train_changes_weights():
    """A client step should actually move the weights and report sample count."""
    torch.manual_seed(0)
    x = torch.randn(20, 1, 28, 28)
    y = torch.randint(0, 10, (20,))
    loader = DataLoader(TensorDataset(x, y), batch_size=10, shuffle=True)
    model = SmallCNN()
    client = Client(0, loader, torch.device("cpu"))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    state, n, loss = client.local_train(model, before, local_epochs=1, lr=0.1, momentum=0.0)
    assert n == 20
    assert isinstance(loss, float)
    moved = any(not torch.allclose(before[k], state[k]) for k in before
                if before[k].dtype.is_floating_point)
    assert moved, "weights did not change after local training"


def test_end_to_end_synthetic_runs():
    """Full federated loop runs on synthetic data and returns valid accuracy."""
    cfg = {
        "dataset": "mnist", "num_clients": 3, "client_fraction": 1.0,
        "rounds": 2, "local_epochs": 1, "batch_size": 10, "lr": 0.01,
        "momentum": 0.0, "seed": 0, "device": "cpu", "eval_batch_size": 32,
        "download": False, "model": {"name": "smallcnn"},
    }
    # Monkeypatch the dataset loader to avoid any download.
    import src.engine as engine

    def fake_load(name, data_root, download):
        x = torch.randn(60, 1, 28, 28)
        y = torch.randint(0, 10, (60,))
        ds = TensorDataset(x, y)
        meta = {"in_channels": 1, "num_classes": 10, "input_size": 28}
        return ds, ds, meta

    orig = engine.load_datasets
    engine.load_datasets = fake_load
    try:
        history, model, privacy = run_federated(cfg, log=lambda *a, **k: None)
    finally:
        engine.load_datasets = orig

    assert privacy is None  # no DP in this config
    assert len(history) == 2
    assert all(0.0 <= h["test_acc"] <= 1.0 for h in history)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
