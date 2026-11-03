"""Reusable attack-under-defense evaluation (shared by the eval scripts)."""

import numpy as np
import torch

from src import attack, defenses, metrics
from src.models import build_model, dlg_init
from src.utils import set_seed

DEFENSES = ["none", "dp", "mask"]
ATTACKERS = ["basic", "adaptive", "cosine"]


def build_attack_model(meta, seed):
    """A freshly-initialised DLG target (sigmoid LeNet + uniform init)."""
    set_seed(seed)
    model = dlg_init(build_model({
        "name": "lenet", "in_channels": meta["in_channels"],
        "num_classes": meta["num_classes"], "input_size": meta["input_size"]}))
    model.eval()
    return model


def defended_gradient(true_grads, defense, params):
    """Apply a defense to the shared gradient before the attacker sees it."""
    if defense == "none":
        return true_grads
    if defense == "dp":
        return defenses.privatize_gradient(
            true_grads, max_grad_norm=params["dp_clip"],
            noise_multiplier=params["dp_noise"], seed=params["def_seed"])
    if defense == "mask":
        masks = defenses.pairwise_masks(
            true_grads, params["num_clients"],
            base_seed=params["def_seed"], scale=params["mask_scale"])
        return defenses.apply_mask(true_grads, masks[0])
    raise ValueError(defense)


def run_attacker(model, observed, x, label, meta, defense, params):
    """Reconstruct `x` from the observed gradient with the configured attacker."""
    attacker = params["attacker"]
    if attacker == "cosine":
        return attack.reconstruct_cosine(
            model, observed, x.shape, label,
            iters=params["cosine_iters"], seed=params["attack_seed"])
    kwargs = {}
    if attacker == "adaptive":
        kwargs["label"] = label
        kwargs["clip"] = params["dp_clip"] if defense == "dp" else None
    return attack.reconstruct(
        model, observed, input_shape=x.shape, num_classes=meta["num_classes"],
        mode=params["mode"], iters=params["iters"], seed=params["attack_seed"], **kwargs)


def attack_model(model, image, label, meta, defense, params):
    """Attack `model` on one image under one defense."""
    x, y = image.unsqueeze(0), torch.tensor([label])
    true_grads = attack.compute_gradient(model, x, y)
    observed = defended_gradient(true_grads, defense, params)
    recon, info = run_attacker(model, observed, x, label, meta, defense, params)
    # An optimizer that blows up has failed; score it as a blank image.
    diverged = not bool(torch.isfinite(recon).all())
    recon = torch.nan_to_num(recon, nan=0.0, posinf=1.0, neginf=0.0)
    m = metrics.all_metrics(recon[0], image)
    return recon[0].clamp(0, 1), {
        "defense": defense, "attacker": params["attacker"], "diverged": diverged, **m,
        "final_grad_loss": info["final_loss"],
        "recovered_label": info["label"], "true_label": label,
        "label_correct": int(info["label"] == label),
    }


def attack_image(image, label, meta, defense, params):
    """Attack a freshly-initialised model on one image under one defense."""
    model = build_attack_model(meta, params["model_seed"])
    return attack_model(model, image, label, meta, defense, params)


def default_params(**overrides):
    p = {
        "attacker": "basic", "mode": "idlg", "iters": 300, "cosine_iters": 3000,
        "model_seed": 42, "attack_seed": 0, "def_seed": 42,
        "dp_clip": 1.0, "dp_noise": 1.0,
        "num_clients": 5, "mask_scale": 1.0,
    }
    p.update(overrides)
    return p


def aggregate(records, success_ssim=0.9):
    """Summarise a list of per-image metric dicts (one defense)."""
    def stat(key):
        vals = np.array([r[key] for r in records], dtype=float)
        return {"mean": float(vals.mean()), "std": float(vals.std()),
                "min": float(vals.min()), "max": float(vals.max())}
    ssim_vals = np.array([r["ssim"] for r in records])
    return {
        "n": len(records),
        "ssim": stat("ssim"), "psnr": stat("psnr"), "mse": stat("mse"),
        "attack_success_rate": float((ssim_vals >= success_ssim).mean()),
        "label_accuracy": float(np.mean([r["label_correct"] for r in records])),
    }
