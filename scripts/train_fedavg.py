"""CLI entry point for the M2 baseline FedAvg run."""

import argparse
import json
import os
import sys

# Allow running as a plain script (`python scripts/train_fedavg.py`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.engine import run_federated


def parse_args():
    p = argparse.ArgumentParser(description="Baseline FedAvg.")
    p.add_argument("--config", default=None, help="YAML config path")
    p.add_argument("--dataset", choices=["mnist", "cifar10"])
    p.add_argument("--num-clients", type=int, dest="num_clients")
    p.add_argument("--client-fraction", type=float, dest="client_fraction")
    p.add_argument("--rounds", type=int)
    p.add_argument("--local-epochs", type=int, dest="local_epochs")
    p.add_argument("--batch-size", type=int, dest="batch_size")
    p.add_argument("--lr", type=float)
    p.add_argument("--momentum", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--limit-train", type=int, dest="limit_train")
    p.add_argument("--partition", choices=["iid", "dirichlet"])
    p.add_argument("--alpha", type=float, dest="dirichlet_alpha",
                   help="Dirichlet concentration (smaller = more label skew)")
    p.add_argument("--mask", action="store_true", help="aggregate via pairwise masking")
    p.add_argument("--mask-scale", type=float, default=1.0, help="mask std dev")
    p.add_argument("--dp", action="store_true", help="enable DP-SGD (Opacus)")
    p.add_argument("--dp-noise", type=float, default=1.0, help="DP noise multiplier")
    p.add_argument("--dp-clip", type=float, default=1.0, help="DP max grad norm")
    p.add_argument("--out", default="results/fedavg_history.json")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def save_plot(history, path, dataset):
    """Save accuracy and loss curves (skips silently if matplotlib absent)."""
    try:
        from src.plotting import SERIES, plt, style
    except ImportError:
        return None
    style()
    rounds = [h["round"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2), sharex=True)
    ax1.plot(rounds, [h["test_acc"] for h in history], "o-", color=SERIES[0])
    ax1.set_xlabel("communication round")
    ax1.set_title("Test accuracy", loc="left")
    ax2.plot(rounds, [h["test_loss"] for h in history], "o-", color=SERIES[0])
    ax2.set_xlabel("communication round")
    ax2.set_title("Test loss", loc="left")
    fig.suptitle(f"FedAvg on {dataset.upper()}: convergence")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path


def main():
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("config", "out", "no_plot", "dp", "dp_noise", "dp_clip",
                              "mask", "mask_scale")}
    if args.mask:
        overrides["masking"] = {"scale": args.mask_scale}
    if args.dp:
        overrides["dp"] = {"noise_multiplier": args.dp_noise, "max_grad_norm": args.dp_clip}
    cfg = load_config(args.config, overrides)

    print("=== Federated config ===")
    print(json.dumps({k: v for k, v in cfg.items() if k != "model"}, indent=2))

    history, _, privacy = run_federated(cfg)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": cfg, "history": history, "privacy": privacy}, f, indent=2)
    print(f"\nSaved history -> {args.out}")

    best = max(history, key=lambda h: h["test_acc"])
    print(f"Final test accuracy: {history[-1]['test_acc']:.4f}")
    print(f"Best test accuracy : {best['test_acc']:.4f} (round {best['round']})")
    if privacy:
        print(f"Privacy budget   : (epsilon={privacy['epsilon']:.3f}, "
              f"delta={privacy['delta']:.0e}) per client")

    if not args.no_plot:
        plot_path = save_plot(history, os.path.splitext(args.out)[0] + ".png", cfg["dataset"])
        if plot_path:
            print(f"Saved plot    -> {plot_path}")


if __name__ == "__main__":
    main()
