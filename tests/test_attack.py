"""Tests for M3 (attack), M4 (defenses), and M5 (metrics)."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import attack, defenses, metrics
from src.models import build_model, dlg_init
from src.utils import set_seed


def _toy_model():
    set_seed(0)
    return dlg_init(build_model({
        "name": "lenet", "in_channels": 1, "num_classes": 10, "input_size": 28})).eval()


def test_metrics_identity():
    img = torch.rand(1, 28, 28)
    m = metrics.all_metrics(img, img)
    assert abs(m["ssim"] - 1.0) < 1e-4
    assert m["mse"] < 1e-6
    assert m["psnr"] > 60


def test_metrics_worse_for_noise():
    img = torch.rand(1, 28, 28)
    noisy = (img + 0.5 * torch.randn_like(img))
    m = metrics.all_metrics(noisy, img)
    assert m["ssim"] < 0.9 and m["mse"] > 0


def test_pairwise_masks_cancel():
    refs = [torch.zeros(4, 3), torch.zeros(5)]
    masks = defenses.pairwise_masks(refs, num_clients=5, base_seed=1)
    for layer in range(len(refs)):
        total = sum(masks[k][layer] for k in range(5))
        assert torch.allclose(total, torch.zeros_like(refs[layer]), atol=1e-5)


def test_masking_preserves_aggregate():
    # Sum of masked updates must equal sum of true updates (training unaffected).
    torch.manual_seed(0)
    updates = [[torch.randn(6), torch.randn(2, 2)] for _ in range(4)]
    refs = updates[0]
    masks = defenses.pairwise_masks(refs, num_clients=4, base_seed=7)
    masked = [defenses.apply_mask(u, masks[k]) for k, u in enumerate(updates)]
    for layer in range(2):
        true_sum = sum(u[layer] for u in updates)
        masked_sum = sum(mu[layer] for mu in masked)
        assert torch.allclose(true_sum, masked_sum, atol=1e-5)


def test_dp_clips_and_noises():
    torch.manual_seed(0)
    grads = [torch.randn(10) * 5]  # large norm
    out = defenses.privatize_gradient(grads, max_grad_norm=1.0, noise_multiplier=0.0)
    # With zero noise, the result is the clipped gradient: norm ~= clip bound.
    assert defenses.global_norm(out).item() <= 1.0 + 1e-4
    noisy = defenses.privatize_gradient(grads, max_grad_norm=1.0, noise_multiplier=1.0)
    assert not torch.allclose(noisy[0], out[0])  # noise was added


def test_idlg_label_recovery():
    model = _toy_model()
    x = torch.rand(1, 1, 28, 28)
    y = torch.tensor([3])
    grads = attack.compute_gradient(model, x, y)
    assert attack.extract_label(grads) == 3


def test_dlg_reconstructs_without_defense():
    """The core M3 claim: the attack recovers the image from the gradient."""
    model = _toy_model()
    torch.manual_seed(1)
    img = torch.rand(1, 28, 28)
    x, y = img.unsqueeze(0), torch.tensor([5])
    grads = attack.compute_gradient(model, x, y)
    recon, info = attack.reconstruct(model, grads, x.shape, 10,
                                     mode="idlg", iters=200, seed=0)
    m = metrics.all_metrics(recon[0], img)
    assert m["ssim"] > 0.95, f"reconstruction failed: SSIM={m['ssim']}"


def test_adaptive_attacker_defeats_clipping_alone():
    """Clipping without noise is no defense once the attacker models the clip."""
    model = _toy_model()
    torch.manual_seed(1)
    img = torch.rand(1, 28, 28)
    x, y = img.unsqueeze(0), torch.tensor([5])
    grads = attack.compute_gradient(model, x, y)
    clip = defenses.global_norm(grads).item() / 5
    clipped = defenses.privatize_gradient(grads, max_grad_norm=clip, noise_multiplier=0.0)
    naive, _ = attack.reconstruct(model, clipped, x.shape, 10, iters=200, seed=0, label=5)
    aware, _ = attack.reconstruct(model, clipped, x.shape, 10, iters=200, seed=0, label=5,
                                  clip=clip)
    assert metrics.all_metrics(aware[0], img)["ssim"] > 0.9
    assert metrics.all_metrics(aware[0], img)["ssim"] > metrics.all_metrics(naive[0], img)["ssim"]


def test_cosine_attack_reconstructs():
    model = _toy_model()
    torch.manual_seed(1)
    img = torch.rand(1, 28, 28)
    x, y = img.unsqueeze(0), torch.tensor([5])
    grads = attack.compute_gradient(model, x, y)
    recon, _ = attack.reconstruct_cosine(model, grads, x.shape, 5, iters=1500, seed=0)
    assert metrics.all_metrics(recon[0], img)["ssim"] > 0.8


def test_defense_degrades_reconstruction():
    """The core M4 claim: defenses make the reconstruction much worse."""
    model = _toy_model()
    torch.manual_seed(2)
    img = torch.rand(1, 28, 28)
    x, y = img.unsqueeze(0), torch.tensor([5])
    grads = attack.compute_gradient(model, x, y)

    clean, _ = attack.reconstruct(model, grads, x.shape, 10, mode="idlg", iters=200, seed=0)
    dp_grads = defenses.privatize_gradient(grads, max_grad_norm=1.0, noise_multiplier=1.0)
    defended, _ = attack.reconstruct(model, dp_grads, x.shape, 10, mode="idlg", iters=200, seed=0)

    clean_ssim = metrics.all_metrics(clean[0], img)["ssim"]
    defended_ssim = metrics.all_metrics(defended[0], img)["ssim"]
    assert defended_ssim < clean_ssim - 0.3, \
        f"defense did not degrade reconstruction (clean={clean_ssim}, dp={defended_ssim})"


def test_epsilon_monotonic_in_noise():
    """More noise must yield a smaller epsilon (stronger privacy)."""
    from src.privacy_accounting import compute_epsilon
    e_low = compute_epsilon(noise_multiplier=0.5, sample_rate=0.01, steps=200)
    e_high = compute_epsilon(noise_multiplier=2.0, sample_rate=0.01, steps=200)
    assert e_high < e_low
    assert 0 < e_high < float("inf")


def test_epsilon_grows_with_steps():
    """More training steps spend more privacy budget (larger epsilon)."""
    from src.privacy_accounting import compute_epsilon
    few = compute_epsilon(1.0, 0.01, 50)
    many = compute_epsilon(1.0, 0.01, 500)
    assert many > few


def test_epsilon_infinite_without_noise():
    from src.privacy_accounting import compute_epsilon
    assert compute_epsilon(0.0, 0.01, 100) == float("inf")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
