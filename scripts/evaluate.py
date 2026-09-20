"""Paper-grade evaluation: aggregate attack metrics over many images."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_sample
from src.evaluation import DEFENSES, aggregate, attack_image, default_params


def load_images(dataset, n):
    images = []
    for i in range(n):
        img, label, meta = get_sample(dataset, i)
        images.append((img, label))
    return images, meta


def run_defense_over_images(defense, images, meta, params, log):
    records = []
    for k, (img, label) in enumerate(images):
        _, m = attack_image(img, label, meta, defense, params)
        records.append(m)
        log(f"  [{defense}] image {k + 1}/{len(images)} "
            f"SSIM={m['ssim']:.3f} label={'ok' if m['label_correct'] else 'x'}")
    return records


def markdown_table(aggregates, success_ssim):
    lines = [
        f"| Defense | SSIM (mean±std) ↓ | PSNR ↓ | MSE ↑ | Attack success (SSIM≥{success_ssim}) ↓ | Label acc ↓ |",
        "|---|---|---|---|---|---|",
    ]
    for d, a in aggregates.items():
        lines.append(
            f"| {d} | {a['ssim']['mean']:.3f} ± {a['ssim']['std']:.3f} "
            f"| {a['psnr']['mean']:.2f} | {a['mse']['mean']:.4f} "
            f"| {a['attack_success_rate'] * 100:.0f}% | {a['label_accuracy'] * 100:.0f}% |")
    return "\n".join(lines)


def save_bar_chart(aggregates, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    defs = list(aggregates.keys())
    means = [aggregates[d]["ssim"]["mean"] for d in defs]
    stds = [aggregates[d]["ssim"]["std"] for d in defs]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(defs, means, yerr=stds, capsize=6,
           color=["tab:red", "tab:green", "tab:blue"][:len(defs)])
    ax.set_ylabel("reconstruction SSIM (lower = more private)")
    ax.set_title("Attack effectiveness vs defense (mean ± std)")
    ax.axhline(0, color="k", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return path


def save_sweep_chart(sweep, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    noises = [s["noise"] for s in sweep]
    means = [s["agg"]["ssim"]["mean"] for s in sweep]
    stds = [s["agg"]["ssim"]["std"] for s in sweep]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.errorbar(noises, means, yerr=stds, marker="o", capsize=4)
    ax.set_xlabel("DP noise multiplier σ")
    ax.set_ylabel("reconstruction SSIM ↓")
    ax.set_title("Privacy strength vs DP noise")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p.add_argument("--num-images", type=int, default=25)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--mode", choices=["dlg", "idlg"], default="idlg")
    p.add_argument("--dp-noise", type=float, default=0.1)
    p.add_argument("--dp-clip", type=float, default=1.0)
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument("--success-ssim", type=float, default=0.9)
    p.add_argument("--seeds", nargs="+", type=int, default=[0],
                   help="independent trial seeds; each varies victim+attack+defense RNG")
    p.add_argument("--dp-sweep", nargs="+", type=float, default=None,
                   help="if set, sweep these DP noise levels instead of the 3-way table")
    p.add_argument("--out", default="results/evaluation")
    args = p.parse_args()

    images, meta = load_images(args.dataset, args.num_images)
    base = default_params(mode=args.mode, iters=args.iters, dp_clip=args.dp_clip,
                          dp_noise=args.dp_noise, num_clients=args.num_clients)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"Evaluating on {len(images)} {args.dataset} images "
          f"({args.mode}, {args.iters} iters)\n")

    if args.dp_sweep:
        # Privacy-strength curve: how SSIM falls as DP noise rises.
        none_records = run_defense_over_images("none", images, meta, base, print)
        sweep = [{"noise": 0.0, "agg": aggregate(none_records, args.success_ssim)}]
        for sigma in args.dp_sweep:
            params = default_params(mode=args.mode, iters=args.iters,
                                    dp_clip=args.dp_clip, dp_noise=sigma)
            print(f"  -- DP noise σ={sigma} --")
            recs = run_defense_over_images("dp", images, meta, params, print)
            sweep.append({"noise": sigma, "agg": aggregate(recs, args.success_ssim)})
        with open(args.out + "_sweep.json", "w") as f:
            json.dump(sweep, f, indent=2)
        print("\n=== DP noise sweep (SSIM ↓ as σ ↑) ===")
        for s in sweep:
            print(f"  σ={s['noise']:<5} SSIM={s['agg']['ssim']['mean']:.3f} "
                  f"± {s['agg']['ssim']['std']:.3f}  "
                  f"success={s['agg']['attack_success_rate']*100:.0f}%")
        fig = save_sweep_chart(sweep, args.out + "_sweep.png")
        print(f"\nSaved sweep -> {args.out}_sweep.json" + (f", {fig}" if fig else ""))
        return

    # Main 3-way table, repeated over independent seeds for statistical rigor.
    all_records = {d: [] for d in DEFENSES}
    per_seed_ssim = {d: [] for d in DEFENSES}  # each seed's mean SSIM
    for s in args.seeds:
        if len(args.seeds) > 1:
            print(f"-- seed {s} --")
        params = default_params(mode=args.mode, iters=args.iters, dp_clip=args.dp_clip,
                                dp_noise=args.dp_noise, num_clients=args.num_clients,
                                model_seed=s, attack_seed=s, def_seed=s)
        for defense in DEFENSES:
            recs = run_defense_over_images(defense, images, meta, params, print)
            all_records[defense].extend(recs)
            per_seed_ssim[defense].append(aggregate(recs, args.success_ssim)["ssim"]["mean"])

    aggregates = {d: aggregate(all_records[d], args.success_ssim) for d in DEFENSES}

    table = markdown_table(aggregates, args.success_ssim)
    n_runs = len(images) * len(args.seeds)
    print(f"\n=== Aggregate results (n = {len(images)} images x "
          f"{len(args.seeds)} seeds = {n_runs} runs) ===")
    print(table)

    if len(args.seeds) > 1:
        print(f"\nAcross-seed stability (mean SSIM ± std over {len(args.seeds)} seeds):")
        for d in DEFENSES:
            vals = np.array(per_seed_ssim[d])
            print(f"  {d:<6} {vals.mean():.4f} ± {vals.std():.4f}")

    with open(args.out + ".json", "w") as f:
        json.dump({"args": vars(args), "aggregates": aggregates,
                   "per_seed_ssim": per_seed_ssim, "records": all_records}, f, indent=2)
    with open(args.out + ".md", "w") as f:
        f.write(f"# Attack-vs-defense evaluation "
                f"(n={len(images)} {args.dataset} images x {len(args.seeds)} seeds)\n\n")
        f.write(table + "\n")
    fig = save_bar_chart(aggregates, args.out + ".png")
    print(f"\nSaved -> {args.out}.json, {args.out}.md" + (f", {fig}" if fig else ""))


if __name__ == "__main__":
    main()
