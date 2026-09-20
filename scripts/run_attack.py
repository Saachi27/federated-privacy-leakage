"""Demo: reconstruct a private image from a gradient."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import attack, metrics
from src.data import get_sample
from src.models import build_model, dlg_init
from src.utils import set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--mode", choices=["dlg", "idlg"], default="idlg")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/attack_reconstruction.png")
    args = p.parse_args()

    image, label, meta = get_sample(args.dataset, args.index)

    set_seed(args.seed)
    model = dlg_init(build_model({
        "name": "lenet", "in_channels": meta["in_channels"],
        "num_classes": meta["num_classes"], "input_size": meta["input_size"]}))
    model.eval()

    x, y = image.unsqueeze(0), torch.tensor([label])
    observed = attack.compute_gradient(model, x, y)
    print(f"Attacking {args.dataset} index={args.index} (true label={label}) "
          f"with {args.mode}, {args.iters} iters...")

    recon, info = attack.reconstruct(
        model, observed, input_shape=x.shape, num_classes=meta["num_classes"],
        mode=args.mode, iters=args.iters, seed=args.seed,
        log_every=max(1, args.iters // 5))

    m = metrics.all_metrics(recon[0], image)
    print(f"\nRecovered label : {info['label']}  (true {label})")
    print(f"Final grad loss : {info['final_loss']:.3e}")
    print(f"MSE {m['mse']:.4f} | PSNR {m['psnr']:.2f} dB | SSIM {m['ssim']:.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available — skipping figure)")
        return

    rec = recon[0].clamp(0, 1).detach().cpu()
    orig = image.cpu()
    fig, axes = plt.subplots(1, 2, figsize=(5, 3))
    for ax, arr, title in [(axes[0], orig, "original"),
                           (axes[1], rec, f"reconstructed\nSSIM={m['ssim']:.3f}")]:
        if arr.shape[0] == 1:
            ax.imshow(arr[0], cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(arr.permute(1, 2, 0).clamp(0, 1))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Deep Leakage from Gradients")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"Saved figure -> {args.out}")


if __name__ == "__main__":
    main()
