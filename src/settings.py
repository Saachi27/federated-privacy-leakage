"""Attack settings for the study: which model the attacker targets and what it sees."""

import copy

import torch
import torch.nn.functional as F

from src import attack
from src.evaluation import build_attack_model

SETTINGS = {
    "mnist-fresh": {"dataset": "mnist", "model_path": None, "update": "gradient"},
    "mnist-trained": {"dataset": "mnist", "model_path": "results/models/mnist_lenet_r20_s{seed}.pt",
                      "update": "gradient"},
    "cifar10-fresh": {"dataset": "cifar10", "model_path": None, "update": "gradient"},
    "mnist-fedavg": {"dataset": "mnist", "model_path": None, "update": "fedavg",
                     "local_steps": 5, "local_lr": 0.01},
}


def build_target(meta, seed, model_path=None):
    """The model under attack: fresh for this seed, or the trained checkpoint for this seed."""
    model = build_attack_model(meta, seed)
    if model_path:
        model.load_state_dict(torch.load(model_path.format(seed=seed), map_location="cpu"))
        model.eval()
    return model


def fedavg_update(model, x, y, local_steps, local_lr):
    """A client's FedAvg upload on (x, y), scaled to look like a mean gradient."""
    local = copy.deepcopy(model)
    local.train()
    opt = torch.optim.SGD(local.parameters(), lr=local_lr)
    for _ in range(local_steps):
        opt.zero_grad()
        F.cross_entropy(local(x), y).backward()
        opt.step()
    scale = local_lr * local_steps
    return [(p0 - p1).detach() / scale for p0, p1 in zip(model.parameters(), local.parameters())]


def per_example_gradients(model, images, labels):
    return [attack.compute_gradient(model, x[None], torch.tensor([y])) for x, y in zip(images, labels)]
