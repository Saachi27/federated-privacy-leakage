"""Config defaults + YAML loading + CLI overrides."""

import yaml

DEFAULTS = {
    "dataset": "mnist",
    "data_root": "./data",
    "download": True,
    "num_clients": 10,
    "client_fraction": 1.0,
    "rounds": 20,
    "local_epochs": 1,
    "batch_size": 64,
    "lr": 0.01,
    "momentum": 0.9,
    "seed": 42,
    "device": "auto",
    "eval_batch_size": 256,
    "limit_train": None,  # cap training samples (None = all); for quick runs
    "dp": None,  # set to {max_grad_norm, noise_multiplier} for
    "model": {"name": "smallcnn"},
}


def load_config(path=None, overrides=None):
    """Merge DEFAULTS < YAML file < CLI overrides (later wins)."""
    cfg = dict(DEFAULTS)
    if path:
        with open(path) as f:
            cfg.update(yaml.safe_load(f) or {})
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg
