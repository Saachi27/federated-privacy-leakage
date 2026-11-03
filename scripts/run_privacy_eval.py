"""Attack one image under each defense, then score."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_sample
from src.evaluation import ATTACKERS, DEFENSES, attack_image, default_params


def run_one(defense, image, label, meta, args):
    params = default_params(
        attacker=args.attacker, mode=args.mode, iters=args.iters,
        model_seed=args.seed, def_seed=args.seed,
        dp_clip=args.dp_clip, dp_noise=args.dp_noise,
        num_clients=args.num_clients, mask_scale=args.mask_scale)
    recon, m = attack_image(image, label, meta, defense, params)
    return recon, m


def save_figure(image, recons, results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    def show(ax, img, title):
        arr = img.detach().cpu()
        if arr.shape[0] == 1:
            ax.imshow(arr[0], cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(arr.permute(1, 2, 0).clamp(0, 1))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    n = 1 + len(recons)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.8))
    show(axes[0], image, "original")
    for ax, (defense, rec) in zip(axes[1:], recons.items()):
        r = results[defense]
        show(ax, rec, f"{defense}\nSSIM={r['ssim']:.2f}  MSE={r['mse']:.3f}")
    fig.suptitle("Gradient-inversion attack: reconstruction vs defense", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--mode", choices=["dlg", "idlg"], default="idlg")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-clients", type=int, default=5, help="for masking")
    p.add_argument("--mask-scale", type=float, default=1.0)
    p.add_argument("--dp-clip", type=float, default=1.0)
    p.add_argument("--dp-noise", type=float, default=1.0)
    p.add_argument("--attacker", choices=ATTACKERS, default="adaptive")
    p.add_argument("--defenses", nargs="+", default=DEFENSES, choices=DEFENSES)
    p.add_argument("--out", default="results/privacy_eval")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    image, label, meta = get_sample(args.dataset, args.index)
    print(f"Target: {args.dataset} index={args.index} true label={label} "
          f"| attacker={args.attacker} iters={args.iters}\n")

    recons, results = {}, {}
    for defense in args.defenses:
        print(f"[{defense}] reconstructing...")
        rec, res = run_one(defense, image, label, meta, args)
        recons[defense], results[defense] = rec, res

    print("\n=== M5 results: reconstruction quality (lower SSIM/PSNR = better privacy) ===")
    print(f"{'defense':<8} {'SSIM↓':>8} {'PSNR↓':>8} {'MSE↑':>10} {'grad-loss':>12} {'label':>6}")
    for d in args.defenses:
        r = results[d]
        print(f"{d:<8} {r['ssim']:>8.3f} {r['psnr']:>8.2f} {r['mse']:>10.4f} "
              f"{r['final_grad_loss']:>12.2e} {r['recovered_label']:>6}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump({"args": vars(args), "true_label": label, "results": results}, f, indent=2)
    print(f"\nSaved metrics -> {args.out}.json")
    fig = save_figure(image, recons, results, args.out + ".png")
    if fig:
        print(f"Saved figure  -> {fig}")


if __name__ == "__main__":
    main()
