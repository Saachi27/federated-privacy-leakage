"""Attack what secure aggregation leaves visible: the sum over K examples."""

import argparse
import json
import multiprocessing
import os
import sys
import time
import warnings

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate import load_images
from src import attack, metrics, recon_store
from src.evaluation import build_attack_model


def match_images(recon, originals):
    """Pair reconstructions with originals to maximize SSIM; return one record per image."""
    return match_and_order(recon, originals)[0]


def match_and_order(recon, originals):
    """Pair reconstructions with originals to maximize SSIM."""
    k = len(originals)
    recon = torch.nan_to_num(recon, nan=0.0, posinf=1.0, neginf=0.0)
    scores = [[metrics.all_metrics(recon[i], originals[j]) for j in range(k)] for i in range(k)]
    rows, cols = linear_sum_assignment(np.array([[s["ssim"] for s in row] for row in scores]), maximize=True)
    by_original = {int(j): int(i) for i, j in zip(rows, cols)}
    order = [by_original[j] for j in range(k)]
    return [scores[order[j]][j] for j in range(k)], recon[order]


def _init_worker():
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")


def _run(task):
    k, attacker, seed, group, images, labels, meta, iters, cosine_iters, cosine_tv = task
    start = time.time()
    model = build_attack_model(meta, seed)
    x, y = torch.stack(images), torch.tensor(labels)
    observed = attack.compute_gradient(model, x, y)
    if attacker == "cosine":
        recon, info = attack.reconstruct_cosine(model, observed, x.shape, labels, iters=cosine_iters,
                                                tv_weight=cosine_tv, seed=seed)
    else:
        recon, info = attack.reconstruct(model, observed, x.shape, meta["num_classes"],
                                         label=labels, iters=iters, seed=seed)
    records, ordered = match_and_order(recon, images)
    return {"k": k, "attacker": attacker, "seed": seed, "group": group,
            "ssim": [r["ssim"] for r in records], "ssim_zeropad": [r["ssim_zeropad"] for r in records],
            "psnr": [r["psnr"] for r in records], "mse": [r["mse"] for r in records],
            "final_loss": info["final_loss"], "seconds": time.time() - start,
            "recon": recon_store.to_array(ordered)}


def make_tasks(args, seed, images, meta):
    """Split the same images into sums of K, in an order that depends on the seed."""
    order = np.random.default_rng(seed).permutation(len(images))
    tasks = []
    for k in args.ks:
        for group in range(len(images) // k):
            chosen = [images[int(i)] for i in order[group * k:(group + 1) * k]]
            for attacker in args.attackers:
                tasks.append((k, attacker, seed, group, [c[0] for c in chosen], [c[1] for c in chosen],
                              meta, args.iters, args.cosine_iters, args.cosine_tv))
    return tasks


def summarize(results, args):
    """Report per-seed means, so the interval reflects seeds and not just images."""
    from scipy import stats
    lines = ["| K | Attacker | Sums | Images | Mean SSIM (95% CI over seeds) | Success (SSIM≥0.9) | Best image per sum |",
             "|---|---|---|---|---|---|---|"]
    for k in args.ks:
        for attacker in args.attackers:
            rows = [r for r in results if r["k"] == k and r["attacker"] == attacker]
            if not rows:
                continue
            by_seed = {}
            for r in rows:
                by_seed.setdefault(r["seed"], []).extend(r["ssim"])
            means = np.array([np.mean(v) for v in by_seed.values()])
            half = stats.t.ppf(0.975, len(means) - 1) * stats.sem(means) if len(means) > 1 else float("nan")
            ssim = np.concatenate([r["ssim"] for r in rows])
            best = np.mean([max(r["ssim"]) for r in rows])
            lines.append(f"| {k} | {attacker} | {len(rows)} | {len(ssim)} | {means.mean():.3f} ± {half:.3f} "
                         f"| {(ssim >= 0.9).mean() * 100:.0f}% | {best:.3f} |")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--attackers", nargs="+", choices=["basic", "cosine"], default=["basic", "cosine"])
    p.add_argument("--num-images", type=int, default=32)
    p.add_argument("--first-image", type=int, default=50, help="skip the images used to choose settings")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--cosine-iters", type=int, default=3000)
    # Chosen on images 25-49 with scripts/validate_cosine.py; those images are not used here.
    p.add_argument("--cosine-tv", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="results/runs/sum")
    args = p.parse_args()

    images, meta = load_images("mnist", args.first_image + args.num_images)
    images = images[args.first_image:]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results = []
    for seed in args.seeds:
        path = f"{args.out}_s{seed}.json"  # finished seeds are skipped
        if os.path.exists(path):
            with open(path) as f:
                results += json.load(f)["records"]
            continue
        tasks = make_tasks(args, seed, images, meta)
        records = []
        with multiprocessing.Pool(args.workers, initializer=_init_worker) as pool:
            for n, rec in enumerate(pool.imap_unordered(_run, tasks, chunksize=1)):
                records.append(rec)
                print(f"seed {seed}: {n + 1}/{len(tasks)} done (K={rec['k']} {rec['attacker']}, "
                      f"{rec['seconds']:.0f}s)", flush=True)
        recon_store.save(f"{args.out}_s{seed}_recon.npz", recon_store.pop_recons(records))
        with open(path, "w") as f:
            json.dump({"args": vars(args), "records": records}, f, indent=2)
        results += records

    lines = summarize(results, args)
    print("\n".join(lines))
    with open(f"{args.out}_summary.md", "w") as f:
        f.write(f"# Attack on the sum over K examples (MNIST, fresh LeNet, {len(args.seeds)} seeds)\n\n")
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
