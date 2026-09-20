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


def extract_label(observed_grads):
    """IDLG label recovery."""
    return int(torch.argmin(observed_grads[-1]).item())


def reconstruct(model, observed_grads, input_shape, num_classes,
                mode="idlg", label=None, iters=300, lr=1.0, seed=0,
                device=None, log_every=0, log=print):
    """Reconstruct the input from an observed gradient."""
    device = device or next(model.parameters()).device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dummy_x = torch.randn(input_shape, generator=gen).to(device).requires_grad_(True)

    if mode == "idlg":
        label = extract_label(observed_grads) if label is None else label
        label_t = torch.tensor([label], device=device)
        params = [dummy_x]
    elif mode == "dlg":
        # Optimise the label jointly as free logits (soft label via softmax).
        dummy_label = torch.randn((1, num_classes), generator=gen).to(device).requires_grad_(True)
        params = [dummy_x, dummy_label]
    else:
        raise ValueError(f"Unknown mode: {mode}")

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
