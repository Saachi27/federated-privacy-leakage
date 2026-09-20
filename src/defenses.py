"""Protections against gradient leakage."""

import torch


# 1.
def _pair_seed(base_seed, i, j):
    # Deterministic shared seed for the unordered pair {i, j}.
    return (base_seed * 1_000_003 + min(i, j) * 1009 + max(i, j)) % (2 ** 31)


def pairwise_masks(reference_tensors, num_clients, base_seed=0, scale=1.0):
    """Build each client's total additive mask."""
    masks = [[torch.zeros_like(t) for t in reference_tensors] for _ in range(num_clients)]
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            gen = torch.Generator(device="cpu").manual_seed(_pair_seed(base_seed, i, j))
            for layer, ref in enumerate(reference_tensors):
                r = torch.randn(ref.shape, generator=gen) * scale
                masks[i][layer] += r  # client i adds
                masks[j][layer] -= r  # client j subtracts -> cancels in sum
    return masks


def apply_mask(tensors, mask):
    """Return tensors + mask (the client's masked upload)."""
    return [t + m for t, m in zip(tensors, mask)]


# 2.
def global_norm(tensors):
    return torch.sqrt(sum((t ** 2).sum() for t in tensors))


def privatize_gradient(tensors, max_grad_norm=1.0, noise_multiplier=1.0, seed=0):
    """Clip the gradient to `max_grad_norm`, then add Gaussian noise."""
    norm = global_norm(tensors)
    clip_coef = min(1.0, (max_grad_norm / (norm + 1e-10)).item())
    gen = torch.Generator(device="cpu").manual_seed(seed)
    std = noise_multiplier * max_grad_norm
    out = []
    for t in tensors:
        clipped = t * clip_coef
        noise = torch.randn(t.shape, generator=gen).to(t.device) * std
        out.append(clipped + noise)
    return out
