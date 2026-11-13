"""Train the attack target with FedAvg and save it, for the "trained model" setting."""

import argparse
import json
import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.engine import run_federated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    cfg = load_config(None, {
        "dataset": args.dataset, "model": {"name": "lenet", "dlg_init": True},
        "num_clients": args.num_clients, "rounds": args.rounds, "seed": args.seed,
        "device": "cpu", "normalize": False, "lr": args.lr,
    })
    history, model, _ = run_federated(cfg)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.cpu().state_dict(), args.out)
    info = {"args": vars(args), "test_acc": history[-1]["test_acc"], "history": history}
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"Saved -> {args.out} (test accuracy {history[-1]['test_acc']:.4f})")


if __name__ == "__main__":
    main()
