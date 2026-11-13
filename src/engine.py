"""The federated training loop: wires clients, server, and evaluation together."""

import random

from .client import Client
from .data import (
    load_datasets, make_client_loaders, make_test_loader, partition_dirichlet,
    partition_iid,
)
from .models import build_model, dlg_init
from .server import average_states, secure_average_states, select_clients
from .utils import evaluate, get_device, set_seed


def resolve_model_cfg(cfg, meta):
    """Fill model in/out dims from dataset metadata unless overridden."""
    model_cfg = dict(cfg.get("model", {}))
    model_cfg.setdefault("name", "smallcnn")
    model_cfg.setdefault("in_channels", meta["in_channels"])
    model_cfg.setdefault("num_classes", meta["num_classes"])
    model_cfg.setdefault("input_size", meta["input_size"])
    return model_cfg


def worst_client_budget(clients, delta=1e-5):
    """The largest per-client (epsilon, delta) budget among clients that took DP steps."""
    from .privacy_accounting import summarize
    budgets = [summarize(c.dp_noise, c.dp_sample_rate, c.dp_steps, delta=delta)
               for c in clients if c.dp_steps > 0]
    return max(budgets, key=lambda b: b["epsilon"])


def run_federated(cfg, log=print, on_round=None):
    """Run FedAvg per the config dict."""
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    log(f"[setup] device={device} | dataset={cfg['dataset']} | "
        f"clients={cfg['num_clients']} fraction={cfg['client_fraction']}")

    train_ds, test_ds, meta = load_datasets(
        cfg["dataset"], cfg.get("data_root", "./data"), cfg.get("download", True),
        cfg.get("normalize", True))

    # Optional caps to keep smoke runs fast.
    n_train = len(train_ds)
    if cfg.get("limit_train"):
        n_train = min(n_train, cfg["limit_train"])
    if cfg.get("partition", "iid") == "dirichlet":
        client_indices = partition_dirichlet(
            train_ds.targets[:n_train], cfg["num_clients"],
            cfg.get("dirichlet_alpha", 0.5), cfg["seed"])
    else:
        client_indices = partition_iid(n_train, cfg["num_clients"], cfg["seed"])
    client_loaders = make_client_loaders(train_ds, client_indices, cfg["batch_size"])

    test_loader = make_test_loader(test_ds, cfg.get("eval_batch_size", 256))

    model_cfg = resolve_model_cfg(cfg, meta)
    global_model = build_model(model_cfg)
    if model_cfg.get("dlg_init"):
        dlg_init(global_model)
    global_model = global_model.to(device)
    global_state = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
    clients = [Client(i, ld, device) for i, ld in enumerate(client_loaders)]

    dp = cfg.get("dp")
    if dp:
        log(f"[setup] differential privacy ON "
            f"(noise={dp['noise_multiplier']}, clip={dp['max_grad_norm']})")

    masking = cfg.get("masking")
    if masking:
        log(f"[setup] pairwise masking ON (scale={masking['scale']})")

    if on_round:
        on_round(0, global_model, client_indices, train_ds)

    rng = random.Random(cfg["seed"])
    history = []
    for rnd in range(1, cfg["rounds"] + 1):
        selected = select_clients(len(clients), cfg["client_fraction"], rng)
        states, counts, losses = [], [], []
        for cid in selected:
            state, n, loss = clients[cid].local_train(
                global_model, global_state,
                cfg["local_epochs"], cfg["lr"], cfg.get("momentum", 0.0), dp=dp)
            states.append(state)
            counts.append(n)
            losses.append(loss)

        if masking:
            # Fresh seeds every round: reusing a mask would let the server cancel it by differencing a
            global_state, _ = secure_average_states(
                states, counts, base_seed=cfg["seed"] * 100_003 + rnd,
                scale=masking["scale"])
        else:
            global_state = average_states(states, counts)
        global_model.load_state_dict(global_state)

        acc, test_loss = evaluate(global_model, test_loader, device)
        # Clients' training losses are computed on private data with no noise
        train_loss = None if dp else sum(losses) / len(losses)
        history.append({
            "round": rnd, "test_acc": acc, "test_loss": test_loss,
            "train_loss": train_loss, "clients": selected,
        })
        shown = "n/a (private)" if train_loss is None else f"{train_loss:.4f}"
        log(f"[round {rnd:3d}/{cfg['rounds']}] "
            f"train_loss={shown} test_loss={test_loss:.4f} "
            f"test_acc={acc:.4f}")
        if on_round:
            on_round(rnd, global_model, client_indices, train_ds)

    privacy = None
    if dp:
        # Report the worst per-client privacy budget.
        privacy = worst_client_budget(clients, cfg.get("dp_delta", 1e-5))
        log(f"[privacy] DP budget per client: epsilon={privacy['epsilon']:.3f}, "
            f"delta={privacy['delta']:.0e}  "
            f"(sigma={privacy['noise_multiplier']}, steps={privacy['steps']})")

    return history, global_model, privacy
