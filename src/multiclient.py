"""Multi-client, multi-step releases under secure aggregation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from src import attack
from src.settings import fedavg_update


def assert_no_batchnorm(model):
    """The sum = batch link (and attacker 2's simulation) assumes no BatchNorm."""
    for m in model.modules():
        assert not isinstance(m, nn.modules.batchnorm._BatchNorm), "model must not use BatchNorm"


def _check_split(images, split):
    num_clients, batch = split
    assert num_clients >= 1 and batch >= 1, split
    assert len(images) == num_clients * batch, f"{len(images)} images do not fit split {split}"
    return num_clients, batch


def make_release(model, images, labels, split, local_steps, local_lr):
    """What the server sees: sum_c n_c * delta_c / (N * eta * E), as a list of tensors."""
    assert_no_batchnorm(model)
    num_clients, batch = _check_split(images, split)
    n = len(images)
    total = None
    for c in range(num_clients):
        part = slice(c * batch, (c + 1) * batch)
        # Fedavg_update returns delta_c / (eta * E); weight it by n_c / N.
        upload = fedavg_update(model, images[part], labels[part], local_steps, local_lr)
        upload = [u * (batch / n) for u in upload]
        total = upload if total is None else [t + u for t, u in zip(total, upload)]
    return total


def cosine(a, b):
    """Cosine between two updates given as lists of tensors."""
    dot = sum((x * y).sum() for x, y in zip(a, b))
    na = torch.sqrt(sum((x ** 2).sum() for x in a))
    nb = torch.sqrt(sum((y ** 2).sum() for y in b))
    return float(dot / (na * nb + 1e-30))


def norm(a):
    return float(torch.sqrt(sum((x ** 2).sum() for x in a)))


def drift(model, images, labels, local_steps, local_lr):
    """(drift cosine, norm ratio) of the N x 1 and 1 x N releases on the same images."""
    n = len(images)
    multi = make_release(model, images, labels, (n, 1), local_steps, local_lr)
    single = make_release(model, images, labels, (1, n), local_steps, local_lr)
    return cosine(multi, single), norm(multi) / norm(single)


def drift_cosine(model, images, labels, local_steps, local_lr):
    """The drift cosine alone (see drift)."""
    return drift(model, images, labels, local_steps, local_lr)[0]


def simulate_release(model, dummy, labels, split, local_steps, local_lr):
    """Make_release on dummy images, differentiable with respect to `dummy`."""
    num_clients, batch = _check_split(dummy, split)
    n = len(dummy)
    names = [name for name, _ in model.named_parameters()]
    start = [p.detach().requires_grad_(True) for p in model.parameters()]
    scale = batch / (n * local_lr * local_steps)
    total = None
    for c in range(num_clients):
        part = slice(c * batch, (c + 1) * batch)
        x, y = dummy[part], labels[part]
        params = start
        for _ in range(local_steps):
            out = functional_call(model, dict(zip(names, params)), (x,))
            grads = torch.autograd.grad(F.cross_entropy(out, y), params, create_graph=True)
            params = [p - local_lr * g for p, g in zip(params, grads)]
        upload = [(p0 - p1) * scale for p0, p1 in zip(start, params)]
        total = upload if total is None else [t + u for t, u in zip(total, upload)]
    return total


def total_variation(x):
    """The TV prior of attack.reconstruct_cosine (mean absolute neighbour difference)."""
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()


def cosine_tv_loss(simulated, observed, dummy, tv_weight):
    """1 - cos(simulated, observed) + tv_weight * TV(dummy), as in attack.reconstruct_cosine."""
    observed_norm = torch.sqrt(sum((g ** 2).sum() for g in observed))
    dot = sum((s * o).sum() for s, o in zip(simulated, observed))
    sim_norm = torch.sqrt(sum((s ** 2).sum() for s in simulated))
    return 1 - dot / (sim_norm * observed_norm + 1e-10) + tv_weight * total_variation(dummy)


def unrolled_objective(model, dummy, labels, observed, split, local_steps, local_lr, tv_weight):
    """Attacker 2's loss for these dummy images (differentiable)."""
    simulated = simulate_release(model, dummy, labels, split, local_steps, local_lr)
    return cosine_tv_loss(simulated, observed, dummy, tv_weight)


def _reconstruct_unrolled_once(model, observed, shape, labels, split, local_steps, local_lr,
                               iters, lr, tv_weight, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # Same start as attack.reconstruct_cosine for the same seed.
    dummy = torch.rand(shape, generator=gen).requires_grad_(True)
    optimizer = torch.optim.Adam([dummy], lr=lr)
    milestones = [iters * 3 // 8, iters * 5 // 8, iters * 7 // 8]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    for _ in range(iters):
        optimizer.zero_grad()
        loss = unrolled_objective(model, dummy, labels, observed, split, local_steps, local_lr, tv_weight)
        (grad,) = torch.autograd.grad(loss, dummy)
        dummy.grad = grad.sign()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            dummy.clamp_(0, 1)
    final = unrolled_objective(model, dummy, labels, observed, split, local_steps, local_lr, tv_weight)
    return dummy.detach(), float(final.detach())


def reconstruct_unrolled(model, observed, shape, labels, split, local_steps, local_lr,
                         iters=3000, lr=0.1, tv_weight=1e-4, seed=0, restarts=1):
    """Attacker 2."""
    assert_no_batchnorm(model)
    labels_t = torch.as_tensor(labels).view(-1)
    best, losses = None, []
    for r in range(restarts):
        recon, loss = _reconstruct_unrolled_once(model, observed, shape, labels_t, split, local_steps,
                                                 local_lr, iters, lr, tv_weight, seed + r)
        losses.append(loss)
        if best is None or _better(loss, losses[best[0]]):
            best = (r, recon)
    return best[1], {"label": list(labels_t.tolist()), "final_loss": losses[best[0]],
                     "restart_losses": losses, "best_restart": best[0],
                     "restart_seeds": [seed + r for r in range(restarts)]}


def _key(loss):
    """A NaN or infinite loss counts as +inf."""
    return loss if loss == loss and loss != float("-inf") else float("inf")


def _better(a, b):
    """A is strictly lower than b, NaN and inf counted as +inf; so a tie keeps the earlier restart."""
    return _key(a) < _key(b)


def l2_loss(model, dummy, labels, observed):
    """The L-BFGS attack's loss: squared distance between gradients."""
    grads = attack.compute_gradient(model, dummy, torch.as_tensor(labels).view(-1))
    return float(sum(((g - o) ** 2).sum() for g, o in zip(grads, observed)))


def cosine_loss(model, dummy, labels, observed, tv_weight):
    """The cosine attack's loss at these images (attack.reconstruct_cosine's objective)."""
    labels_t = torch.as_tensor(labels).view(-1)
    model.zero_grad()
    grads = torch.autograd.grad(F.cross_entropy(model(dummy), labels_t), model.parameters())
    return float(cosine_tv_loss(grads, observed, dummy, tv_weight))


def attack_one(model, observed, shape, labels, kind, num_classes=10, iters=300, cosine_iters=3000,
               tv_weight=1e-4, seed=0, restarts=1):
    """Attacker 1: treat the release as a mean gradient and run src/attack.py."""
    labels = list(torch.as_tensor(labels).view(-1).tolist())
    best, losses = None, []
    for r in range(restarts):
        if kind == "lbfgs":
            recon, _ = attack.reconstruct(model, observed, shape, num_classes, label=labels,
                                          iters=iters, seed=seed + r, restarts=1)
            loss = l2_loss(model, recon, labels, observed) if torch.isfinite(recon).all() else float("nan")
        elif kind == "cosine":
            recon, _ = attack.reconstruct_cosine(model, observed, shape, labels, iters=cosine_iters,
                                                 tv_weight=tv_weight, seed=seed + r)
            loss = cosine_loss(model, recon, labels, observed, tv_weight)
        else:
            raise ValueError(kind)
        losses.append(loss)
        if best is None or _better(loss, losses[best[0]]):
            best = (r, recon)
    return best[1], {"label": labels, "final_loss": losses[best[0]],
                     "restart_losses": losses, "best_restart": best[0],
                     "restart_seeds": [seed + r for r in range(restarts)]}

