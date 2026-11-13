"""RQ4 control: attack one example with the same relative noise as the K-sum."""

import argparse
import json
import multiprocessing
import os
import sys
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.attack_noise_sum import noisy_mean_gradient
from scripts.evaluate import load_images
from src import attack, metrics
from src.evaluation import build_attack_model


def global_norm(grads):
    return float(torch.sqrt(sum((g ** 2).sum() for g in grads)))


def matched_sigmas(per_example, clip, sigma):
    """Sigma'_i for each example so its relative noise equals the K-sum's (after clipping)."""
    clipped = []
    for grads in per_example:
        coef = min(1.0, clip / (global_norm(grads) + 1e-10))
        clipped.append([g * coef for g in grads])
    sum_norm = global_norm([sum(parts) for parts in zip(*clipped)])
    return [sigma * global_norm(c) / (sum_norm + 1e-10) for c in clipped]


def _init_worker():
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")


def _run(task):
    k, attacker, seed, group, sigma, images, labels, meta, clip, iters, cosine_iters, cosine_tv = task
    start = time.time()
    model = build_attack_model(meta, seed)
    per_example = [attack.compute_gradient(model, x[None], torch.tensor([y])) for x, y in zip(images, labels)]
    # The z of attack_noise_sum.py for this (seed, K, group), so the noise direction matches.
    gen = torch.Generator().manual_seed(seed * 1_000_003 + k * 1009 + group)
    z = [torch.randn(g.shape, generator=gen) for g in per_example[0]]
    sigma_eff = matched_sigmas(per_example, clip, sigma)
    shape = (1,) + tuple(images[0].shape)
    ssim, psnr, mse, losses = [], [], [], []
    for grads, x, y, s in zip(per_example, images, labels, sigma_eff):
        observed, _ = noisy_mean_gradient([grads], clip, s, z=z)
        if attacker == "cosine":
            recon, info = attack.reconstruct_cosine(model, observed, shape, [y], iters=cosine_iters,
                                                    tv_weight=cosine_tv, seed=seed)
        else:
            recon, info = attack.reconstruct(model, observed, shape, meta["num_classes"],
                                             label=[y], iters=iters, seed=seed)
        m = metrics.all_metrics(torch.nan_to_num(recon[0], nan=0.0, posinf=1.0, neginf=0.0), x)
        ssim.append(m["ssim"]), psnr.append(m["psnr"]), mse.append(m["mse"]), losses.append(info["final_loss"])
    return {"k": k, "attacker": attacker, "seed": seed, "group": group, "sigma": sigma,
            "sigma_eff": sigma_eff, "ssim": ssim, "psnr": psnr, "mse": mse, "final_loss": losses,
            "seconds": time.time() - start}


def make_tasks(args, seed, images, meta):
    """Same seed-dependent grouping as attack_sum.py and attack_noise_sum.py."""
    order = np.random.default_rng(seed).permutation(len(images))
    tasks = []
    for k in args.ks:
        for group in range(len(images) // k):
            chosen = [images[int(i)] for i in order[group * k:(group + 1) * k]]
            for sigma in args.sigmas:
                for attacker in args.attackers:
                    tasks.append((k, attacker, seed, group, sigma, [c[0] for c in chosen],
                                  [c[1] for c in chosen], meta, args.clip, args.iters,
                                  args.cosine_iters, args.cosine_tv))
    tasks.sort(key=lambda t: -(t[0] * (10 if t[1] == "cosine" else 1)))  # slow tasks first
    return tasks


def summarize(results, args):
    from scipy import stats
    lines = ["| K matched | sigma | Attacker | Images | Mean sigma' | Mean SSIM (95% CI over seeds) "
             "| Success (SSIM≥0.9) |", "|---|---|---|---|---|---|---|"]
    for k in args.ks:
        for sigma in args.sigmas:
            for attacker in args.attackers:
                rows = [r for r in results if r["k"] == k and r["sigma"] == sigma and r["attacker"] == attacker]
                if not rows:
                    continue
                by_seed = {}
                for r in rows:
                    by_seed.setdefault(r["seed"], []).extend(r["ssim"])
                means = np.array([np.mean(v) for v in by_seed.values()])
                half = stats.t.ppf(0.975, len(means) - 1) * stats.sem(means) if len(means) > 1 else float("nan")
                ssim = np.concatenate([r["ssim"] for r in rows])
                eff = np.concatenate([r["sigma_eff"] for r in rows])
                lines.append(f"| {k} | {sigma:g} | {attacker} | {len(ssim)} | {eff.mean():.3g} | "
                             f"{means.mean():.3f} ± {half:.3f} | {(ssim >= 0.9).mean() * 100:.0f}% |")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ks", nargs="+", type=int, default=[2, 4, 8])
    p.add_argument("--sigmas", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3, 3e-3])
    p.add_argument("--clip", type=float, default=30.0, help="must match attack_noise_sum.py")
    p.add_argument("--attackers", nargs="+", choices=["basic", "cosine"], default=["basic", "cosine"])
    p.add_argument("--num-images", type=int, default=16)
    p.add_argument("--first-image", type=int, default=50)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--cosine-iters", type=int, default=3000)
    p.add_argument("--cosine-tv", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="results/runs/matched")
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
                print(f"seed {seed}: {n + 1}/{len(tasks)} done (K={rec['k']} sigma={rec['sigma']:g} "
                      f"{rec['attacker']}, SSIM {np.mean(rec['ssim']):.3f}, {rec['seconds']:.0f}s)", flush=True)
        with open(path, "w") as f:
            json.dump({"args": vars(args), "records": records}, f, indent=2)
        results += records

    lines = ["# Matched-SNR control for RQ4 (MNIST, fresh LeNet)", "",
             "Each example attacked alone, with noise scaled to the K-sum's signal-to-noise.", ""]
    lines += summarize(results, args)
    with open(f"{args.out}_summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
