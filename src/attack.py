"""Gradient-inversion attack (DLG / iDLG)."""

import torch
import torch.nn.functional as F


def compute_gradient(model, x, y, criterion=None):
    """The 'shared gradient': ∇θ L for a single example (detached copy)."""
    criterion = criterion or torch.nn.CrossEntropyLoss()
    model.zero_grad()
    loss = criterion(model(x), y)
    grads = torch.autograd.grad(loss, model.parameters())
    return [g.detach().clone() for g in grads]


def clip_by_global_norm(grads, clip):
    """Scale gradients down to global norm `clip` (differentiable)."""
    norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
    return [g * torch.clamp(clip / (norm + 1e-10), max=1.0) for g in grads]


def extract_label(observed_grads):
    """IDLG label recovery."""
    return int(torch.argmin(observed_grads[-1]).item())


def reconstruct(model, observed_grads, input_shape, num_classes,
                mode="idlg", label=None, iters=300, lr=1.0, seed=0,
                device=None, log_every=0, log=print, clip=None, restarts=1):
    """Reconstruct the input from an observed gradient."""
    if restarts < 1:
        raise ValueError(f"restarts must be >= 1, got {restarts}")
    if mode not in ("idlg", "dlg"):
        raise ValueError(f"Unknown mode: {mode}")
    device = device or next(model.parameters()).device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if mode == "idlg":
        label = extract_label(observed_grads) if label is None else label

    best = None
    restart_losses = []
    for r in range(restarts):
        if restarts > 1 and log_every:
            log(f"  [attack] restart {r + 1}/{restarts}")
        x, info = _lbfgs_run(model, observed_grads, input_shape, num_classes, mode, label,
                             iters, lr, gen, device, log_every, log, clip)
        restart_losses.append(info["final_loss"])
        if best is None or _sort_key(info["final_loss"]) < _sort_key(best[1]["final_loss"]):
            best = (x, info, r)
    x, info, r = best
    info["restart_losses"] = restart_losses
    info["best_restart"] = r
    return x, info


def _sort_key(loss):
    return float("inf") if loss != loss else loss  # NaN never wins


def _lbfgs_run(model, observed_grads, input_shape, num_classes, mode, label,
               iters, lr, gen, device, log_every, log, clip):
    """One L-BFGS run from a random start drawn from `gen`."""
    dummy_x = torch.randn(input_shape, generator=gen).to(device).requires_grad_(True)

    if mode == "idlg":
        label_t = torch.as_tensor(label, device=device).view(-1)  # one label per example
        params = [dummy_x]
    else:
        # Optimise the label jointly as free logits (soft label via softmax).
        dummy_label = torch.randn((1, num_classes), generator=gen).to(device).requires_grad_(True)
        params = [dummy_x, dummy_label]

    optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=20)
    history = []

    def closure():
        optimizer.zero_grad()
        pred = model(dummy_x)
        if mode == "idlg":
            dummy_loss = F.cross_entropy(pred, label_t)
        else:
            soft = F.softmax(dummy_label, dim=-1)
            dummy_loss = torch.mean(torch.sum(-soft * F.log_softmax(pred, dim=-1), dim=-1))
        dummy_grads = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
        if clip is not None:
            dummy_grads = clip_by_global_norm(dummy_grads, clip)
        grad_diff = sum(((dg - og) ** 2).sum() for dg, og in zip(dummy_grads, observed_grads))
        grad_diff.backward()
        return grad_diff

    for it in range(iters):
        loss = optimizer.step(closure)
        history.append(float(loss))
        if log_every and (it % log_every == 0 or it == iters - 1):
            log(f"  [attack] iter {it:4d}/{iters}  grad-match loss = {float(loss):.6e}")

    if mode == "dlg":
        label = int(torch.argmax(dummy_label).item())
    return dummy_x.detach(), {"label": label, "final_loss": history[-1], "history": history}


def reconstruct_cosine(model, observed_grads, input_shape, label,
                       iters=3000, lr=0.1, tv_weight=1e-4, seed=0, device=None):
    """Geiping et al."""
    device = device or next(model.parameters()).device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dummy_x = torch.rand(input_shape, generator=gen).to(device).requires_grad_(True)
    label_t = torch.as_tensor(label, device=device).view(-1)  # one label per example
    observed_norm = torch.sqrt(sum((g ** 2).sum() for g in observed_grads))

    optimizer = torch.optim.Adam([dummy_x], lr=lr)
    milestones = [iters * 3 // 8, iters * 5 // 8, iters * 7 // 8]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

    history = []
    for _ in range(iters):
        optimizer.zero_grad()
        dummy_loss = F.cross_entropy(model(dummy_x), label_t)
        dummy_grads = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
        dot = sum((dg * og).sum() for dg, og in zip(dummy_grads, observed_grads))
        dummy_norm = torch.sqrt(sum((dg ** 2).sum() for dg in dummy_grads))
        tv = (dummy_x[..., 1:, :] - dummy_x[..., :-1, :]).abs().mean() + \
             (dummy_x[..., :, 1:] - dummy_x[..., :, :-1]).abs().mean()
        loss = 1 - dot / (dummy_norm * observed_norm + 1e-10) + tv_weight * tv
        loss.backward()
        dummy_x.grad.sign_()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            dummy_x.clamp_(0, 1)
        history.append(float(loss))
    return dummy_x.detach(), {"label": label, "final_loss": history[-1], "history": history}
