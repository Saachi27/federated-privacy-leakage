"""Paper-grade evaluation: aggregate attack metrics over many images."""

import argparse
import json
import multiprocessing
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_datasets
from src.evaluation import ATTACKERS, DEFENSES, aggregate, attack_image, default_params
from src.privacy_accounting import compute_epsilon


def load_images(dataset, n):
    _, test_ds, meta = load_datasets(dataset, normalize=False)
    images = []
    for i in range(n):
        img, label = test_ds[i]
        images.append((img, int(label)))
    return images, meta


def _init_worker():
    # One BLAS thread per process: the workers already use every core.
    import torch
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")


def _run_task(task):
    key, img, label, meta, defense, params = task
    _, m = attack_image(img, label, meta, defense, params)
    return key, m


def run_tasks(tasks, workers):
    """Run attack tasks in parallel, returning {key: [records in task order]}."""
    out = {}
    with multiprocessing.Pool(workers, initializer=_init_worker) as pool:
        for k, (key, m) in enumerate(pool.imap(_run_task, tasks, chunksize=1)):
            out.setdefault(key, []).append(m)
            if (k + 1) % 25 == 0 or k + 1 == len(tasks):
                print(f"  {k + 1}/{len(tasks)} attacks done", flush=True)
    return out


def markdown_table(aggregates, success_ssim):
    lines = [
        f"| Attacker | Defense | SSIM (mean±std) ↓ | PSNR ↓ | MSE ↑ | Attack success (SSIM≥{success_ssim}) ↓ | Label acc ↓ |",
        "|---|---|---|---|---|---|---|",
    ]
    for (attacker, d), a in aggregates.items():
        lines.append(
            f"| {attacker} | {d} | {a['ssim']['mean']:.3f} ± {a['ssim']['std']:.3f} "
            f"| {a['psnr']['mean']:.2f} | {a['mse']['mean']:.4f} "
            f"| {a['attack_success_rate'] * 100:.0f}% | {a['label_accuracy'] * 100:.0f}% |")
    return "\n".join(lines)


def save_bar_chart(aggregates, attackers, path):
    try:
        from src.plotting import SERIES, plt, style
    except ImportError:
        return None
    style()
    x = np.arange(len(DEFENSES))
    width = 0.8 / len(attackers)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for i, attacker in enumerate(attackers):
        means = [aggregates[(attacker, d)]["ssim"]["mean"] for d in DEFENSES]
        stds = [aggregates[(attacker, d)]["ssim"]["std"] for d in DEFENSES]
        ax.bar(x + (i - (len(attackers) - 1) / 2) * width, means, width * 0.92,
               yerr=stds, capsize=3, color=SERIES[i], label=attacker,
               error_kw={"elinewidth": 1, "ecolor": "#52514e"})
    ax.set_xticks(x, ["no defense", "DP noise", "pairwise mask"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("reconstruction SSIM (lower = more private)")
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.legend(title="attacker", ncol=len(attackers), loc="upper right")
    ax.set_title("Attack success by defense (mean ± std)", loc="left")
    fig.savefig(path)
    return path


def save_sweep_chart(sweep, attackers, path):
    try:
        from src.plotting import MARKERS, SERIES, plt, style
    except ImportError:
        return None
    style()
    x = np.arange(len(sweep))
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for i, attacker in enumerate(attackers):
        means = np.array([s["agg"][attacker]["ssim"]["mean"] for s in sweep])
        stds = np.array([s["agg"][attacker]["ssim"]["std"] for s in sweep])
        ax.fill_between(x, means - stds, means + stds, color=SERIES[i], alpha=0.12, linewidth=0)
        ax.plot(x, means, marker=MARKERS[i], color=SERIES[i], label=attacker)
    ax.set_xticks(x, [s["label"] for s in sweep])
    ax.set_xlabel("defense strength (noise multiplier σ, clip norm fixed at 1)")
    ax.set_ylabel("reconstruction SSIM (lower = more private)")
    ax.legend(title="attacker", loc="upper right")
    ax.set_title("How much noise stops the attack?", loc="left")
    fig.savefig(path)
    return path


def run_sweep(args, images, meta):
    """Privacy-strength curve: how SSIM falls as DP noise rises."""
    settings = [("none", 0.0, "none")] + [
        ("dp", sigma, "clip only" if sigma == 0 else f"{sigma:g}") for sigma in args.dp_sweep]
    tasks = []
    for s in args.seeds:
        for defense, sigma, name in settings:
            params = default_params(iters=args.iters, dp_clip=args.dp_clip, dp_noise=sigma,
                                    model_seed=s, attack_seed=s, def_seed=s)
            for attacker in args.attackers:
                for img, label in images:
                    tasks.append(((name, attacker), img, label, meta, defense,
                                  {**params, "attacker": attacker}))
    records = run_tasks(tasks, args.workers)

    sweep = []
    for defense, sigma, name in settings:
        # The DP guarantee of releasing this one clipped, noised gradient.
        eps = compute_epsilon(sigma, 1.0, 1) if defense == "dp" and sigma >= 0.1 else None
        sweep.append({
            "label": name, "noise": sigma, "single_release_epsilon": eps,
            "agg": {a: aggregate(records[(name, a)], args.success_ssim) for a in args.attackers},
        })
    with open(args.out + "_sweep.json", "w") as f:
        json.dump({"args": vars(args), "sweep": sweep}, f, indent=2)

    print("\n=== DP noise sweep (SSIM ↓ as σ ↑) ===")
    lines = [f"# DP noise sweep (n={len(images)} {args.dataset} images x {len(args.seeds)} seeds)\n",
             "| σ | single-release ε | " + " | ".join(
                 f"{a} SSIM / success" for a in args.attackers) + " |",
             "|---|---|" + "---|" * len(args.attackers)]
    for s in sweep:
        cells = [f"{s['agg'][a]['ssim']['mean']:.3f} ± {s['agg'][a]['ssim']['std']:.3f} / "
                 f"{s['agg'][a]['attack_success_rate'] * 100:.0f}%" for a in args.attackers]
        eps = "—" if s["single_release_epsilon"] is None else f"{s['single_release_epsilon']:.2f}"
        lines.append(f"| {s['label']} | {eps} | " + " | ".join(cells) + " |")
    print("\n".join(lines))
    with open(args.out + "_sweep.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    fig = save_sweep_chart(sweep, args.attackers, args.out + "_sweep.png")
    print(f"\nSaved sweep -> {args.out}_sweep.json" + (f", {fig}" if fig else ""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p.add_argument("--num-images", type=int, default=25)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--attackers", nargs="+", choices=ATTACKERS, default=ATTACKERS)
    p.add_argument("--dp-noise", type=float, default=1.0)
    p.add_argument("--dp-clip", type=float, default=1.0)
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument("--success-ssim", type=float, default=0.9)
    p.add_argument("--seeds", nargs="+", type=int, default=[0],
                   help="independent trial seeds; each varies victim+attack+defense RNG")
    p.add_argument("--dp-sweep", nargs="+", type=float, default=None,
                   help="if set, sweep these DP noise levels instead of the 3-way table")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", default="results/evaluation")
    args = p.parse_args()

    images, meta = load_images(args.dataset, args.num_images)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"Evaluating on {len(images)} {args.dataset} images x {len(args.seeds)} seeds, "
          f"attackers={args.attackers}\n", flush=True)

    if args.dp_sweep is not None:
        run_sweep(args, images, meta)
        return

    # Main table, repeated over independent seeds for statistical rigor.
    tasks = []
    for s in args.seeds:
        params = default_params(iters=args.iters, dp_clip=args.dp_clip,
                                dp_noise=args.dp_noise, num_clients=args.num_clients,
                                model_seed=s, attack_seed=s, def_seed=s)
        for attacker in args.attackers:
            for defense in DEFENSES:
                for img, label in images:
                    tasks.append(((attacker, defense, s), img, label, meta, defense,
                                  {**params, "attacker": attacker}))
    by_seed = run_tasks(tasks, args.workers)

    all_records, per_seed_ssim = {}, {}
    for (attacker, defense, s), recs in by_seed.items():
        all_records.setdefault((attacker, defense), []).extend(recs)
        per_seed_ssim.setdefault((attacker, defense), []).append(
            aggregate(recs, args.success_ssim)["ssim"]["mean"])
    aggregates = {k: aggregate(v, args.success_ssim) for k, v in all_records.items()}

    table = markdown_table(aggregates, args.success_ssim)
    n_runs = len(images) * len(args.seeds)
    print(f"\n=== Aggregate results (n = {len(images)} images x "
          f"{len(args.seeds)} seeds = {n_runs} runs per row) ===")
    print(table)

    stability = []
    if len(args.seeds) > 1:
        print(f"\nAcross-seed stability (mean SSIM ± std over {len(args.seeds)} seeds):")
        for (attacker, d), vals in per_seed_ssim.items():
            line = f"{attacker}/{d}: {np.mean(vals):.4f} ± {np.std(vals):.4f}"
            stability.append(line)
            print(f"  {line}")

    flat = lambda d: {f"{a}/{k}": v for (a, k), v in d.items()}
    with open(args.out + ".json", "w") as f:
        json.dump({"args": vars(args), "aggregates": flat(aggregates),
                   "per_seed_ssim": flat(per_seed_ssim), "records": flat(all_records)}, f, indent=2)
    with open(args.out + ".md", "w") as f:
        f.write(f"# Attack-vs-defense evaluation "
                f"(n={len(images)} {args.dataset} images x {len(args.seeds)} seeds, "
                f"DP σ={args.dp_noise}, clip={args.dp_clip})\n\n")
        f.write(table + "\n")
    fig = save_bar_chart(aggregates, args.attackers, args.out + ".png")
    print(f"\nSaved -> {args.out}.json, {args.out}.md" + (f", {fig}" if fig else ""))


if __name__ == "__main__":
    main()
