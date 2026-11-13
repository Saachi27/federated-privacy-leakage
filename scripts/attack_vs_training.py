"""Does the leak survive training?"""

import argparse
import copy
import json
import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import recon_store
from src.config import load_config
from src.data import DATASET_META
from src.engine import run_federated
from src.evaluation import ATTACKERS, aggregate, attack_model, default_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p.add_argument("--model", choices=["lenet", "smallcnn"], default="lenet")
    p.add_argument("--checkpoints", nargs="+", type=int, default=[0, 1, 2, 5, 10, 20])
    p.add_argument("--num-images", type=int, default=10)
    p.add_argument("--attackers", nargs="+", choices=ATTACKERS, default=["basic", "cosine"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--success-ssim", type=float, default=0.9)
    p.add_argument("--cosine-tv", type=float, default=1e-3,
                   help="cosine attacker's TV weight (chosen with scripts/validate_cosine.py)")
    p.add_argument("--out", default="results/runs/attack.json")
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    meta = DATASET_META[args.dataset]
    rows, recons = [], []

    def on_round(rnd, model, client_indices, train_ds):
        if rnd not in args.checkpoints:
            return
        victim = copy.deepcopy(model).cpu().eval()
        images = [train_ds[int(i)] for i in client_indices[0][:args.num_images]]
        for attacker in args.attackers:
            params = default_params(attacker=attacker, attack_seed=args.seed, cosine_tv=args.cosine_tv)
            out = [attack_model(victim, img, int(label), meta, "none", params) for img, label in images]
            records = [{**m, "recon": recon_store.to_array(r)} for r, m in out]
            offset = len(recons)
            recons.extend(recon_store.pop_recons(records))
            for r in records:
                r["recon_id"] += offset
            agg = aggregate(records, args.success_ssim)
            agg["diverged"] = sum(r["diverged"] for r in records)
            rows.append({"round": rnd, "attacker": attacker, "agg": agg, "records": records})
            print(f"[attack] round {rnd:3d} {attacker:<8} SSIM={agg['ssim']['mean']:.3f} "
                  f"success={agg['attack_success_rate'] * 100:.0f}%", flush=True)

    model_cfg = {"name": args.model, "dlg_init": args.model == "lenet"}
    cfg = load_config(None, {
        "dataset": args.dataset, "model": model_cfg, "num_clients": args.num_clients,
        "rounds": max(args.checkpoints), "seed": args.seed, "device": "cpu",
        "normalize": False, "lr": args.lr, "limit_train": args.limit_train,
    })
    history, _, _ = run_federated(cfg, on_round=on_round)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    recon_store.save(os.path.splitext(args.out)[0] + "_recon.npz", recons)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "config": cfg, "history": history, "attacks": rows},
                  f, indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
