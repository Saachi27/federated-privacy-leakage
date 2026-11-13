"""Check the cosine attacker's settings before trusting it as the strong attacker."""

import argparse
import json
import multiprocessing
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate import load_images
from src import attack, metrics
from src.evaluation import build_attack_model

# Name -> overrides of reconstruct_cosine's defaults, plus "restarts"
ARMS = {
    "current (lr 0.1, 3000 iters, TV 1e-4)": {},
    "lr 0.03": {"lr": 0.03},
    "4800 iters": {"iters": 4800},
    "no TV": {"tv_weight": 0.0},
    "TV 1e-3": {"tv_weight": 1e-3},
    "4 restarts": {"restarts": 4},
    "TV 3e-3": {"tv_weight": 3e-3},
    "TV 1e-2": {"tv_weight": 1e-2},
    "TV 1e-3, 4800 iters": {"tv_weight": 1e-3, "iters": 4800},
}


def _init_worker():
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")


def _run(task):
    arm, index, img, label, meta, seed = task
    settings = dict(ARMS[arm])
    restarts = settings.pop("restarts", 1)
    model = build_attack_model(meta, seed)
    x, y = img.unsqueeze(0), torch.tensor([label])
    observed = attack.compute_gradient(model, x, y)
    best = None
    for r in range(restarts):
        recon, info = attack.reconstruct_cosine(model, observed, x.shape, label, seed=seed + r, **settings)
        if best is None or info["final_loss"] < best[1]["final_loss"]:
            best = recon, info
    m = metrics.all_metrics(best[0][0], img)
    return arm, {"image": index, "ssim": m["ssim"], "ssim_zeropad": m["ssim_zeropad"], "psnr": m["psnr"],
                 "final_loss": best[1]["final_loss"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-images", type=int, default=25)
    p.add_argument("--first-image", type=int, default=0, help="use later images as a held-out check")
    p.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS)[:6])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="results/checks/cosine_validation.json")
    args = p.parse_args()

    images, meta = load_images("mnist", args.first_image + args.num_images)
    images = images[args.first_image:]
    tasks = [(arm, i, img, label, meta, args.seed)
             for arm in args.arms for i, (img, label) in enumerate(images)]
    records = {arm: [] for arm in args.arms}
    with multiprocessing.Pool(args.workers, initializer=_init_worker) as pool:
        for k, (arm, rec) in enumerate(pool.imap(_run, tasks, chunksize=1)):
            records[arm].append(rec)
            if (k + 1) % 5 == 0:
                print(f"{k + 1}/{len(tasks)} attacks done", flush=True)

    summary = {}
    for arm, recs in records.items():
        ssim = np.array([r["ssim"] for r in recs])
        summary[arm] = {"n": len(recs), "ssim_mean": float(ssim.mean()), "ssim_std": float(ssim.std()),
                        "success": float((ssim >= 0.9).mean())}
        print(f"{arm:<40} SSIM {ssim.mean():.3f} ± {ssim.std():.3f}  success {(ssim >= 0.9).mean() * 100:.0f}%")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "records": records}, f, indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
