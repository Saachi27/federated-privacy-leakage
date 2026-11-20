"""Choose each setting's clip norm and cosine attacker settings on held-out images."""

import argparse
import json
import multiprocessing
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.attack_noise_sum import _init_worker, _run
from scripts.evaluate import load_images
from src.settings import SETTINGS, build_target, per_example_gradients


OLD_TVS = [1e-4, 1e-3, 3e-3, 1e-2]  # grid behind the existing tuning.json files
TVS = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]  # one decade wider at the top


def grid_edge(grid, best):
    """'low' or 'high' if `best` is the smallest or largest grid value, else None."""
    if best == min(grid):
        return "low"
    if best == max(grid):
        return "high"
    return None


def global_norm(grads):
    return float(torch.sqrt(sum((g ** 2).sum() for g in grads)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setting", choices=list(SETTINGS), required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--norm-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--num-images", type=int, default=16, help="held-out images 0..n-1 for TV tuning")
    p.add_argument("--tvs", nargs="+", type=float, default=TVS)
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 1e-3])
    p.add_argument("--cosine-iters", type=int, default=3000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=None, help="default results/study/<setting>/tuning.json")
    args = p.parse_args()
    warnings.filterwarnings("ignore")
    setting = {"model_path": None, "update": "gradient", "local_steps": 5, "local_lr": 0.01,
               **SETTINGS[args.setting]}
    out = args.out or f"results/study/{args.setting}/tuning.json"
    if os.path.exists(out):
        sys.exit(f"{out} exists; pass a new --out (old tuning results are kept as they are)")
    images, meta = load_images(setting["dataset"], 50)

    # 1.
    norms = {}
    for seed in args.norm_seeds:
        model = build_target(meta, seed, setting["model_path"])
        grads = per_example_gradients(model, [i[0] for i in images], [i[1] for i in images])
        norms[seed] = [global_norm(g) for g in grads]
    all_norms = np.concatenate(list(norms.values()))
    clip = float(1.1 * all_norms.max())
    print(f"per-example gradient norm: median {np.median(all_norms):.3g}, max {all_norms.max():.3g} "
          f"-> clip C = {clip:.3g}", flush=True)

    # 2.
    sigmas = args.sigmas if setting["update"] == "gradient" else [0.0]
    tasks = []
    for seed in args.seeds:
        order = np.random.default_rng(seed).permutation(args.num_images)
        for group in range(args.num_images // args.k):
            chosen = [images[int(i)] for i in order[group * args.k:(group + 1) * args.k]]
            for sigma in sigmas:
                for tv in args.tvs:
                    tasks.append({"k": args.k, "attacker": "cosine", "seed": seed, "group": group,
                                  "sigma": sigma, "images": [c[0] for c in chosen],
                                  "labels": [c[1] for c in chosen], "meta": meta, "clip": clip,
                                  "iters": 300, "cosine_iters": args.cosine_iters, "cosine_tv": tv,
                                  "model_path": setting["model_path"], "update": setting["update"],
                                  "local_steps": setting["local_steps"], "local_lr": setting["local_lr"]})
    scores = {tv: {s: [] for s in sigmas} for tv in args.tvs}
    with multiprocessing.Pool(args.workers, initializer=_init_worker) as pool:
        for n, rec in enumerate(pool.imap_unordered(_run, tasks, chunksize=1)):
            scores[rec["cosine_tv"]][rec["sigma"]].extend(rec["ssim"])
            print(f"{n + 1}/{len(tasks)} done", flush=True)
    table = {tv: {str(s): float(np.mean(v)) for s, v in by_sigma.items()} for tv, by_sigma in scores.items()}
    mean_by_tv = {tv: float(np.mean(list(row.values()))) for tv, row in table.items()}
    best = max(mean_by_tv, key=mean_by_tv.get)
    for tv in args.tvs:
        cells = "  ".join(f"σ={s}: {v:.3f}" for s, v in table[tv].items())
        print(f"TV {tv:g}: {cells}  mean {mean_by_tv[tv]:.3f}" + ("  <- best" if tv == best else ""))

    edge = grid_edge(args.tvs, best)
    if edge:
        print(f"WARNING: best TV {best:g} is at the {edge} edge of the grid; widen --tvs")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"setting": args.setting, "args": vars(args), "clip": clip, "tv_grid_edge": edge,
                   "norms": {"median": float(np.median(all_norms)), "max": float(all_norms.max()),
                             "per_seed": norms},
                   "cosine_tv": best, "ssim_by_tv": table, "mean_by_tv": mean_by_tv}, f, indent=2)
    print(f"Saved -> {out}: clip {clip:.3g}, cosine TV {best:g}")


if __name__ == "__main__":
    main()
