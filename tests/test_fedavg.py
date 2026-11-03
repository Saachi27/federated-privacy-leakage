"""Tests for the FedAvg core."""

import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.client import Client
from src.engine import run_federated
from src.models import SmallCNN
from src.data import partition_dirichlet
from src.server import average_states, secure_average_states, select_clients


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

    def fake_load(name, data_root, download, normalize=True):
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


def test_secure_aggregation_matches_fedavg():
    """Masked aggregation must give FedAvg's result, not any client's weights."""
    torch.manual_seed(0)
    states = [{"w": torch.randn(50), "b": torch.randn(4, 4)} for _ in range(5)]
    counts = [10, 20, 30, 40, 50]
    plain = average_states(states, counts)
    secure, uploads = secure_average_states(states, counts, base_seed=3)
    for key in plain:
        assert torch.allclose(plain[key], secure[key], atol=1e-6), key
    # What the server sees of client 0 is dominated by the mask.
    weighted = states[0]["w"].double() * (counts[0] / sum(counts))
    assert (uploads[0][0] - weighted).abs().mean() > 5 * weighted.abs().mean()


def test_masked_training_matches_baseline():
    """End to end: training with masking stays on the baseline's trajectory."""
    x = torch.randn(60, 1, 28, 28)
    y = torch.randint(0, 10, (60,))
    ds = TensorDataset(x, y)
    meta = {"in_channels": 1, "num_classes": 10, "input_size": 28}
    import src.engine as engine
    orig = engine.load_datasets
    engine.load_datasets = lambda *a, **k: (ds, ds, meta)
    cfg = {
        "dataset": "mnist", "num_clients": 3, "client_fraction": 1.0, "rounds": 3,
        "local_epochs": 1, "batch_size": 10, "lr": 0.05, "momentum": 0.0, "seed": 0,
        "device": "cpu", "model": {"name": "smallcnn"},
    }
    try:
        _, base, _ = run_federated(cfg, log=lambda *a, **k: None)
        _, masked, _ = run_federated({**cfg, "masking": {"scale": 1.0}},
                                     log=lambda *a, **k: None)
    finally:
        engine.load_datasets = orig
    for key, value in base.state_dict().items():
        assert torch.allclose(value, masked.state_dict()[key], atol=1e-4), key


def test_masks_change_every_round():
    """Reused masks could be cancelled by differencing uploads across rounds."""
    states = [{"w": torch.zeros(20)} for _ in range(3)]
    _, first = secure_average_states(states, [1, 1, 1], base_seed=100)
    _, second = secure_average_states(states, [1, 1, 1], base_seed=101)
    assert not torch.allclose(first[0][0], second[0][0])


def test_dirichlet_partition_is_a_skewed_split():
    labels = torch.arange(10).repeat(100).numpy()
    shards = partition_dirichlet(labels, num_clients=5, alpha=0.1, seed=0)
    flat = sorted(i for s in shards for i in s)
    assert flat == list(range(1000)), "every sample lands on exactly one client"
    assert all(len(s) >= 10 for s in shards)
    def top_class_share(shard):
        counts = torch.bincount(torch.tensor(labels[shard]), minlength=10)
        return (counts.max().float() / counts.sum()).item()
    assert max(top_class_share(s) for s in shards) > 0.3, "alpha=0.1 must skew labels"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
