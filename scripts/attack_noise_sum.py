"""Experiment 6 (RQ2): how much Gaussian noise on the sum stops the attack at each K?"""

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

from scripts.attack_sum import match_and_order
from scripts.evaluate import load_images
from src import attack, recon_store
from src.settings import SETTINGS, build_target, fedavg_update, per_example_gradients
from src.privacy_accounting import compute_epsilon

DEFAULT_SIGMAS = [0.0, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1]


def noisy_mean_gradient(per_example, clip, sigma, z=None, seed=0):
    """Clip each example's gradient to norm `clip`, sum, add N(0,(sigma*clip)^2) once, divide by K."""
    k = len(per_example)
    clipped, n_clipped = [], 0
    for grads in per_example:
        norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
        coef = min(1.0, float(clip / (norm + 1e-10)))
        n_clipped += coef < 1.0
        clipped.append([g * coef for g in grads])
    total = [sum(parts) for parts in zip(*clipped)]
    if sigma > 0:
        if z is None:
            gen = torch.Generator().manual_seed(seed)
            z = [torch.randn(t.shape, generator=gen) for t in total]
        total = [t + sigma * clip * n for t, n in zip(total, z)]
    return [t / k for t in total], n_clipped / k


def epsilon_for(sigma, delta=1e-5):
    """Epsilon of one Gaussian-mechanism release (sampling rate 1, one step)."""
    return compute_epsilon(sigma, 1.0, 1, delta) if sigma > 0 else float("inf")


def _init_worker():
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")


def observe(t, model):
    """What the server sees for this task: (observed update, fraction of examples clipped)."""
    k, seed, group, sigma = t["k"], t["seed"], t["group"], t["sigma"]
    if t["update"] == "fedavg":
        # Noise on a multi-step client update protects the whole client (user-level DP)
        assert sigma == 0, "the fedavg setting is run without noise"
        x, y = torch.stack(t["images"]), torch.tensor(t["labels"])
        return fedavg_update(model, x, y, t["local_steps"], t["local_lr"]), 0.0
    per_example = per_example_gradients(model, t["images"], t["labels"])
    # Same z for every sigma at this (seed, K, group).
    gen = torch.Generator().manual_seed(seed * 1_000_003 + k * 1009 + group)
    z = [torch.randn(g.shape, generator=gen) for g in per_example[0]]
    return noisy_mean_gradient(per_example, t["clip"], sigma, z=z)


def _run(t):
    k, attacker, seed, images, labels, meta = t["k"], t["attacker"], t["seed"], t["images"], t["labels"], t["meta"]
    start = time.time()
    model = build_target(meta, seed, t["model_path"])
    observed, frac_clipped = observe(t, model)
    shape = (k,) + tuple(images[0].shape)
    if attacker == "cosine":
        recon, info = attack.reconstruct_cosine(model, observed, shape, labels, iters=t["cosine_iters"],
                                                tv_weight=t["cosine_tv"], seed=seed)
    else:
        recon, info = attack.reconstruct(model, observed, shape, meta["num_classes"],
                                         label=labels, iters=t["iters"], seed=seed)
    group, sigma = t["group"], t["sigma"]
    records, ordered = match_and_order(recon, images)
    return {"k": k, "attacker": attacker, "seed": seed, "group": group, "sigma": sigma,
            "cosine_tv": t["cosine_tv"] if attacker == "cosine" else None,
            "epsilon": epsilon_for(sigma), "frac_clipped": frac_clipped,
            "ssim": [r["ssim"] for r in records], "ssim_zeropad": [r["ssim_zeropad"] for r in records],
            "psnr": [r["psnr"] for r in records], "mse": [r["mse"] for r in records],
            "final_loss": info["final_loss"], "seconds": time.time() - start,
            "recon": recon_store.to_array(ordered)}


def make_tasks(args, seed, images, meta):
    """Same seed-dependent grouping as attack_sum.py, repeated at every sigma."""
    order = np.random.default_rng(seed).permutation(len(images))
    tasks = []
    for k in args.ks:
        for group in range(len(images) // k):
            chosen = [images[int(i)] for i in order[group * k:(group + 1) * k]]
            for sigma in args.sigmas:
                for attacker in args.attackers:
                    tasks.append({"k": k, "attacker": attacker, "seed": seed, "group": group, "sigma": sigma,
                                  "images": [c[0] for c in chosen], "labels": [c[1] for c in chosen],
                                  "meta": meta, "clip": args.clip, "iters": args.iters,
                                  "cosine_iters": args.cosine_iters, "cosine_tv": args.cosine_tv,
                                  "model_path": args.model_path, "update": args.update,
                                  "local_steps": args.local_steps, "local_lr": args.local_lr})
    # Slow tasks first so the pool does not end on one long straggler.
    tasks.sort(key=lambda t: -(t["k"] * (10 if t["attacker"] == "cosine" else 1)))
    return tasks


def summarize(results, args):
    """Per-seed means, so the interval reflects seeds and not just images."""
    from scipy import stats
    lines = [f"Clip C = {args.clip}. Epsilon is for one Gaussian release (delta=1e-5, sampling rate 1, "
             "one step); it is not the DP-SGD training epsilon.", "",
             "| K | sigma | epsilon | Attacker | Images | Mean SSIM (95% CI over seeds) | Success (SSIM≥0.9) "
             "| Mean PSNR | Clipped |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k in args.ks:
        for sigma in args.sigmas:
            for attacker in args.attackers:
                rows = [r for r in results if r["k"] == k and r["attacker"] == attacker and r["sigma"] == sigma]
                if not rows:
                    continue
                by_seed = {}
                for r in rows:
                    by_seed.setdefault(r["seed"], []).extend(r["ssim"])
                means = np.array([np.mean(v) for v in by_seed.values()])
                half = stats.t.ppf(0.975, len(means) - 1) * stats.sem(means) if len(means) > 1 else float("nan")
                ssim = np.concatenate([r["ssim"] for r in rows])
                psnr = np.concatenate([r["psnr"] for r in rows])
                psnr = psnr[np.isfinite(psnr)]
                eps = epsilon_for(sigma)
                eps_s = "∞" if not np.isfinite(eps) else f"{eps:.3g}"
                clipped = np.mean([r["frac_clipped"] for r in rows])
                lines.append(f"| {k} | {sigma:g} | {eps_s} | {attacker} | {len(ssim)} | {means.mean():.3f} ± "
                             f"{half:.3f} | {(ssim >= 0.9).mean() * 100:.0f}% | {psnr.mean():.1f} | "
                             f"{clipped * 100:.0f}% |")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setting", choices=list(SETTINGS), default=None,
                   help="preset for --dataset, --model-path and --update (src/settings.py)")
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p.add_argument("--model-path", default=None, help="trained checkpoint, with {seed}; fresh model if unset")
    p.add_argument("--update", choices=["gradient", "fedavg"], default="gradient")
    p.add_argument("--local-steps", type=int, default=5)
    p.add_argument("--local-lr", type=float, default=0.01)
    p.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--sigmas", nargs="+", type=float, default=DEFAULT_SIGMAS)
    p.add_argument("--clip", type=float, default=30.0, help="per-example clip norm C (see docstring)")
    p.add_argument("--attackers", nargs="+", choices=["basic", "cosine"], default=["basic", "cosine"])
    p.add_argument("--num-images", type=int, default=16)
    p.add_argument("--first-image", type=int, default=50, help="skip the images used to choose settings")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--cosine-iters", type=int, default=3000)
    p.add_argument("--cosine-tv", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="results/runs/noise")
    args = p.parse_args()
    if args.setting:
        for key, value in SETTINGS[args.setting].items():
            setattr(args, key, value)

    images, meta = load_images(args.dataset, args.first_image + args.num_images)
    images = images[args.first_image:]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print("epsilon per sigma (one release, delta=1e-5): " +
          ", ".join(f"{s:g}->{epsilon_for(s):.3g}" for s in args.sigmas), flush=True)
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
        recon_store.save(f"{args.out}_s{seed}_recon.npz", recon_store.pop_recons(records))
        with open(path, "w") as f:
            json.dump({"args": vars(args), "records": records}, f, indent=2)
        results += records

    title = args.setting or f"{args.dataset}, {'trained' if args.model_path else 'fresh'} LeNet, {args.update}"
    lines = [f"# Noise on the sum over K examples ({title})", ""] + summarize(results, args)
    with open(f"{args.out}_summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
