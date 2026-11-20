"""The matched multi-client test."""

import argparse
import glob
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
from src import multiclient as mc, recon_store
from src.data import load_datasets
from src.settings import SETTINGS, build_target

ATTACKER_1 = ["lbfgs", "cosine"]
ATTACKERS = ATTACKER_1 + ["unrolled"]
LR_GRID = [0.01, 0.03, 0.1, 0.3]
DRIFT_TARGET = 0.9
# The cut order: each level drops one more (setting, N) block.
CUTS = [("mnist-trained", 4), ("mnist-fresh", 4), ("mnist-trained", 2)]
# TODO: take these from src/splits.py once it lands (another worker is adding it).
DEFAULT_TUNING_RANGE = (0, 50)
DEFAULT_TARGET_RANGE = (100, 400)


def cells(n, es, splits, attackers):
    """(E, split, attacker) cells for N images."""
    out = []
    for e in es:
        if e == 1:
            out += [(1, (1, n), a) for a in attackers if a in ATTACKER_1]
            continue
        for kind in splits:
            split = (1, n) if kind == "single" else (n, 1)
            out += [(e, split, a) for a in attackers]
    return out


def group_indices(n, num_groups, target_range, seed):
    """Disjoint groups of N test-image indices from the target range."""
    pool = np.arange(*target_range)
    assert num_groups * n <= len(pool), f"{num_groups} groups of {n} need more than {len(pool)} images"
    order = np.random.default_rng(seed).permutation(pool)
    return [[int(i) for i in order[g * n:(g + 1) * n]] for g in range(num_groups)]


def task_path(out, setting, n, e, split, attacker, group):
    return os.path.join(out, setting, f"N{n}", f"E{e}_{split[0]}x{split[1]}_{attacker}_g{group:02d}.json")


def blocks(args):
    """(setting, N) blocks to run, in priority order (the reverse of the cut order)."""
    cut = set(CUTS[:args.cut])
    return [(s, n) for n in sorted(args.ns) for s in args.settings if (s, n) not in cut]


def make_tasks(args):
    tasks = []
    for setting, n in blocks(args):
        lr = local_lr_for(args, setting)
        groups = group_indices(n, args.first_group + args.groups, args.target_range, args.group_seed)
        for g in range(args.first_group, args.first_group + args.groups):
            for e, split, attacker in cells(n, args.es, args.splits, args.attackers):
                tasks.append({
                    "setting": setting, "n": n, "e": e, "split": list(split), "attacker": attacker,
                    "group": g, "checkpoint": args.checkpoints[g % len(args.checkpoints)],
                    "image_indices": groups[g], "local_lr": lr, "cosine_tv": cosine_tv_for(args, setting),
                    "restarts": args.restarts, "iters": args.iters, "cosine_iters": args.cosine_iters,
                    "unrolled_iters": args.unrolled_iters,
                    "path": task_path(args.out, setting, n, e, split, attacker, g)})
    return tasks


def cosine_tv_for(args, setting):
    """--cosine-tv, else the study's tuned value for this setting, else 1e-3."""
    if args.cosine_tv is not None:
        return args.cosine_tv
    # TODO: switch to the wider TV grid when it is tuned.
    path = f"results/study/{setting}/tuning.json"
    if os.path.exists(path):
        with open(path) as f:
            return float(json.load(f)["cosine_tv"])
    return 1e-3


def local_lr_for(args, setting):
    if args.local_lr != "auto":
        return float(args.local_lr)
    path = lr_grid_path(args, setting)
    if not os.path.exists(path):
        sys.exit(f"{path} not found: run --lr-grid first, or pass --local-lr")
    with open(path) as f:
        return float(json.load(f)["chosen_lr"])


def lr_grid_path(args, setting):
    return os.path.join(args.out, setting, "lr_grid.json")


_IMAGES = {}


def load_images(dataset, indices):
    """(image, label) for these test indices, cached per process."""
    if dataset not in _IMAGES:
        _, test_ds, meta = load_datasets(dataset, normalize=False)
        _IMAGES[dataset] = (test_ds, meta)
    test_ds, meta = _IMAGES[dataset]
    pairs = [test_ds[i] for i in indices]
    return torch.stack([p[0] for p in pairs]), torch.tensor([int(p[1]) for p in pairs]), meta


def write_atomic(path, write):
    """Call write(file) on a temporary file, then rename it to path."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def _init_worker(threads):
    torch.set_num_threads(threads)
    warnings.filterwarnings("ignore")


def run_task(t):
    start = time.time()
    setting = SETTINGS[t["setting"]]
    x, y, meta = load_images(setting["dataset"], t["image_indices"])
    model = build_target(meta, t["checkpoint"], setting["model_path"])
    split, e, lr = tuple(t["split"]), t["e"], t["local_lr"]
    observed = mc.make_release(model, x, y, split, e, lr)
    drift = mc.drift_cosine(model, x, y, e, lr)
    seed = 1000 * t["group"]
    if t["attacker"] == "unrolled":
        recon, info = mc.reconstruct_unrolled(model, observed, x.shape, y, split, e, lr,
                                              iters=t["unrolled_iters"], tv_weight=t["cosine_tv"],
                                              seed=seed, restarts=t["restarts"])
    else:
        recon, info = mc.attack_one(model, observed, x.shape, y, t["attacker"], meta["num_classes"],
                                    iters=t["iters"], cosine_iters=t["cosine_iters"],
                                    tv_weight=t["cosine_tv"], seed=seed, restarts=t["restarts"])
    records, ordered = match_and_order(recon, list(x))
    rec = {k: v for k, v in t.items() if k != "path"}
    rec.update({
        "labels": y.tolist(), "drift_cosine": drift,
        "release_norm": float(torch.sqrt(sum((g ** 2).sum() for g in observed))),
        "final_loss": info["final_loss"], "restart_losses": info["restart_losses"],
        "best_restart": info["best_restart"],
        "ssim": [r["ssim"] for r in records], "ssim_zeropad": [r["ssim_zeropad"] for r in records],
        "psnr": [r["psnr"] for r in records], "mse": [r["mse"] for r in records],
        "recon_id": 0, "recon_file": os.path.basename(t["path"]).replace(".json", "_recon.npz"),
        "seconds": time.time() - start})
    os.makedirs(os.path.dirname(t["path"]), exist_ok=True)
    # The npz goes first; the JSON marks the task as finished.
    write_atomic(t["path"].replace(".json", "_recon.npz"),
                 lambda f: recon_store.save(f, [recon_store.to_array(ordered)]))
    write_atomic(t["path"], lambda f: f.write(json.dumps(rec, indent=2).encode()))
    return rec


def run_grid(args):
    tasks = make_tasks(args)
    todo = [t for t in tasks if not os.path.exists(t["path"])]
    print(f"{len(tasks)} tasks, {len(tasks) - len(todo)} already done, {len(todo)} to run", flush=True)
    if args.max_tasks is not None:
        todo = todo[:args.max_tasks]
    if args.dry_run:
        for t in todo:
            print(t["path"])
        return
    with multiprocessing.Pool(args.workers, initializer=_init_worker, initargs=(args.threads,)) as pool:
        for k, rec in enumerate(pool.imap_unordered(run_task, todo, chunksize=1)):
            print(f"{k + 1}/{len(todo)} {rec['setting']} N={rec['n']} E={rec['e']} "
                  f"{rec['split'][0]}x{rec['split'][1]} {rec['attacker']} g{rec['group']}: "
                  f"SSIM {np.mean(rec['ssim']):.3f}, drift {rec['drift_cosine']:.3f}, "
                  f"losses {['%.3g' % v for v in rec['restart_losses']]}, {rec['seconds']:.0f}s", flush=True)


def lr_grid(args):
    """Median drift cosine at N = 2, E = 5 on tuning images, per learning rate."""
    for setting in args.settings:
        path = lr_grid_path(args, setting)
        if os.path.exists(path) and not args.force:
            print(f"skip {path}")
            continue
        start = time.time()
        meta_setting = SETTINGS[setting]
        pairs = group_indices(2, (args.tuning_range[1] - args.tuning_range[0]) // 2, args.tuning_range,
                              args.group_seed)
        drifts = {lr: [] for lr in args.lrs}
        for g, idx in enumerate(pairs):
            x, y, meta = load_images(meta_setting["dataset"], idx)
            model = build_target(meta, args.checkpoints[g % len(args.checkpoints)], meta_setting["model_path"])
            for lr in args.lrs:
                drifts[lr].append(mc.drift_cosine(model, x, y, args.lr_grid_steps, lr))
        medians = {lr: float(np.median(v)) for lr, v in drifts.items()}
        passing = [lr for lr in sorted(args.lrs) if medians[lr] <= DRIFT_TARGET]
        chosen = passing[0] if passing else max(args.lrs)
        result = {"setting": setting, "n": 2, "local_steps": args.lr_grid_steps,
                  "tuning_range": list(args.tuning_range), "checkpoints": args.checkpoints,
                  "rule": f"smallest lr with median drift cosine <= {DRIFT_TARGET}, else the largest",
                  "medians": {str(k): v for k, v in medians.items()},
                  "drifts": {str(k): v for k, v in drifts.items()},
                  "chosen_lr": chosen, "at_grid_edge": not passing, "seconds": time.time() - start}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_atomic(path, lambda f: f.write(json.dumps(result, indent=2).encode()))
        print(f"{setting}: median drift " + ", ".join(f"lr {k:g} -> {v:.3f}" for k, v in medians.items()) +
              f"; chosen lr {chosen:g}" + (" (grid edge)" if not passing else ""), flush=True)


def summary(args):
    rows = {}
    for path in sorted(glob.glob(os.path.join(args.out, "*", "N*", "E*_g*.json"))):
        with open(path) as f:
            r = json.load(f)
        key = (r["setting"], r["n"], r["e"], f"{r['split'][0]}x{r['split'][1]}", r["attacker"])
        rows.setdefault(key, []).append(r)
    lines = ["| Setting | N | E | Split | Attacker | Releases | Mean SSIM | Any image SSIM≥0.9 "
             "| Median drift | Mean s/release |", "|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(rows):
        rs = rows[key]
        ssim = np.mean([np.mean(r["ssim"]) for r in rs])
        anyp = np.mean([max(r["ssim"]) >= 0.9 for r in rs])
        lines.append(f"| {' | '.join(map(str, key))} | {len(rs)} | {ssim:.3f} | {anyp * 100:.0f}% | "
                     f"{np.median([r['drift_cosine'] for r in rs]):.3f} | "
                     f"{np.mean([r['seconds'] for r in rs]):.0f} |")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--settings", nargs="+", choices=["mnist-fresh", "mnist-trained"],
                   default=["mnist-fresh", "mnist-trained"])
    p.add_argument("--ns", nargs="+", type=int, default=[2, 4])
    p.add_argument("--es", nargs="+", type=int, default=[1, 5])
    p.add_argument("--splits", nargs="+", choices=["single", "multi"], default=["single", "multi"],
                   help="single = 1 x N, multi = N x 1 (E > 1 only)")
    p.add_argument("--attackers", nargs="+", choices=ATTACKERS, default=ATTACKERS)
    p.add_argument("--cut", type=int, default=0, choices=range(len(CUTS) + 1),
                   help="apply the cut order up to this level (1: drop trained N=4, "
                        "2: also fresh N=4, 3: also trained N=2)")
    p.add_argument("--groups", type=int, default=30, help="releases per cell")
    p.add_argument("--first-group", type=int, default=0)
    p.add_argument("--checkpoints", nargs="+", type=int, default=[0, 1, 2],
                   help="model seeds; group g uses checkpoints[g %% len]")
    p.add_argument("--target-range", nargs=2, type=int, default=list(DEFAULT_TARGET_RANGE),
                   metavar=("START", "END"), help="test-image indices for releases (END exclusive)")
    p.add_argument("--tuning-range", nargs=2, type=int, default=list(DEFAULT_TUNING_RANGE),
                   metavar=("START", "END"), help="test-image indices for --lr-grid (END exclusive)")
    p.add_argument("--group-seed", type=int, default=0, help="seed of the image draw")
    p.add_argument("--local-lr", default="auto", help="a number, or auto: the --lr-grid pick")
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--iters", type=int, default=300, help="L-BFGS outer steps")
    p.add_argument("--cosine-iters", type=int, default=3000)
    p.add_argument("--unrolled-iters", type=int, default=3000)
    p.add_argument("--cosine-tv", type=float, default=None,
                   help="TV weight for cosine and unrolled (default: the study's tuned value)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    p.add_argument("--max-tasks", type=int, default=None, help="stop after this many new tasks")
    p.add_argument("--dry-run", action="store_true", help="list the tasks that would run")
    p.add_argument("--lr-grid", action="store_true", help="run the learning-rate rule instead of the grid")
    p.add_argument("--lrs", nargs="+", type=float, default=LR_GRID)
    p.add_argument("--lr-grid-steps", type=int, default=5)
    p.add_argument("--force", action="store_true", help="redo --lr-grid even if saved")
    p.add_argument("--summary", action="store_true", help="print a table of finished tasks")
    p.add_argument("--out", default="results/multiclient")
    args = p.parse_args()
    args.target_range, args.tuning_range = tuple(args.target_range), tuple(args.tuning_range)
    lo, hi = sorted([args.target_range, args.tuning_range])
    assert lo[1] <= hi[0], "tuning and target images must not overlap"

    torch.set_num_threads(args.threads)
    if args.lr_grid:
        lr_grid(args)
    elif args.summary:
        summary(args)
    else:
        run_grid(args)


if __name__ == "__main__":
    main()
