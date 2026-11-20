"""Tests for multi-client, multi-step releases and the unrolled attacker (src/multiclient.py)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import attack, metrics, multiclient as mc
from src.models import build_model, dlg_init
from src.utils import set_seed


def _model(double=False):
    set_seed(0)
    model = dlg_init(build_model({"name": "lenet", "in_channels": 1, "num_classes": 10, "input_size": 28}))
    return (model.double() if double else model).eval()


def _batch(n, double=False):
    gen = torch.Generator().manual_seed(1)
    x = torch.rand(n, 1, 28, 28, generator=gen)
    y = torch.randint(0, 10, (n,), generator=gen)
    return (x.double() if double else x), y


def _max_diff(a, b):
    return max(float((u - v).detach().abs().max()) for u, v in zip(a, b))


@pytest.mark.parametrize("n,splits", [(4, [(1, 4), (4, 1)]), (8, [(2, 4), (1, 8)])])
def test_one_step_release_is_the_batch_gradient(n, splits):
    # Float64, so the weight difference does not lose digits to rounding.
    model = _model(double=True)
    x, y = _batch(n, double=True)
    batch = attack.compute_gradient(model, x, y)
    releases = [mc.make_release(model, x, y, split, local_steps=1, local_lr=0.01) for split in splits]
    for release in releases:
        assert _max_diff(release, batch) < 1e-9
    assert _max_diff(releases[0], releases[1]) < 1e-9


def test_drift_cosine_is_one_at_one_step_and_falls_with_local_training():
    model = _model(double=True)
    x, y = _batch(4, double=True)
    assert abs(mc.drift_cosine(model, x, y, local_steps=1, local_lr=0.3) - 1) < 1e-9
    assert mc.drift_cosine(model, x, y, local_steps=5, local_lr=0.3) < 0.9


def test_single_client_release_matches_fedavg_update():
    from src.settings import fedavg_update
    model = _model()
    x, y = _batch(4)
    release = mc.make_release(model, x, y, (1, 4), local_steps=5, local_lr=0.1)
    assert _max_diff(release, fedavg_update(model, x, y, 5, 0.1)) == 0


def test_unrolled_simulation_matches_the_real_release():
    model = _model(double=True)
    x, y = _batch(4, double=True)
    for split in [(4, 1), (2, 2), (1, 4)]:
        real = mc.make_release(model, x, y, split, local_steps=3, local_lr=0.1)
        simulated = mc.simulate_release(model, x, y, split, local_steps=3, local_lr=0.1)
        assert _max_diff(real, simulated) < 1e-9


def test_unrolled_objective_at_one_step_equals_the_cosine_attack_objective():
    # Attacker 1's first loss value is its objective at its starting images
    model = _model()
    x, y = _batch(4)
    observed = mc.make_release(model, x, y, (4, 1), local_steps=1, local_lr=0.01)
    _, info = attack.reconstruct_cosine(model, observed, x.shape, y.tolist(), iters=1, tv_weight=1e-3, seed=7)
    dummy = torch.rand(x.shape, generator=torch.Generator().manual_seed(7))
    for split in [(4, 1), (1, 4), (2, 2)]:
        ours = mc.unrolled_objective(model, dummy, y, observed, split, 1, 0.01, 1e-3)
        assert abs(float(ours) - info["history"][0]) < 1e-5
    assert abs(mc.cosine_loss(model, dummy, y, observed, 1e-3) - info["history"][0]) < 1e-6


def test_restarts_keep_the_lowest_loss():
    model = _model()
    x, y = _batch(2)
    observed = mc.make_release(model, x, y, (2, 1), local_steps=2, local_lr=0.1)
    _, info = mc.reconstruct_unrolled(model, observed, x.shape, y, (2, 1), 2, 0.1, iters=5, restarts=3)
    assert len(info["restart_losses"]) == 3
    assert info["final_loss"] == min(info["restart_losses"])
    _, info = mc.attack_one(model, observed, x.shape, y, "cosine", cosine_iters=5, restarts=2)
    assert info["final_loss"] == min(info["restart_losses"])


def test_batchnorm_models_are_rejected():
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3), torch.nn.BatchNorm2d(2), torch.nn.Flatten(),
                                torch.nn.Linear(2 * 26 * 26, 10))
    x, y = _batch(2)
    with pytest.raises(AssertionError):
        mc.make_release(model, x, y, (2, 1), 1, 0.1)


def test_unrolled_attack_recovers_one_mnist_image():
    pytest.importorskip("torchvision")
    if not os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")):
        pytest.skip("MNIST not downloaded")
    from scripts.evaluate import load_images
    from src.evaluation import build_attack_model
    images, meta = load_images("mnist", 3)
    model = build_attack_model(meta, 0)
    x, y = images[2][0][None], torch.tensor([images[2][1]])
    observed = mc.make_release(model, x, y, (1, 1), local_steps=5, local_lr=0.1)
    recon, _ = mc.reconstruct_unrolled(model, observed, x.shape, y, (1, 1), 5, 0.1, iters=400, tv_weight=1e-3)
    assert metrics.ssim(recon[0], x[0]) > 0.8
