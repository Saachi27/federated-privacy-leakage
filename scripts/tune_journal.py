"""TV tuning and the objective rule for the journal runs."""

import argparse
import json
import multiprocessing
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import attack_multiclient as am
from src import splits

TVS = [1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
EDGE_EXTENSION = {1e-5: 1e-6, 1e-1: 1.0}
NUM_PAIRS = 12
OBJECTIVE_MARGIN = 0.02
# Cell -> (E, splits, attacker)
CELLS = {"T-A": (1, [(1, 2)], "cosine"), "T-B": (5, [(1, 2), (2, 1)], "cosine"),
         "T-C": (5, [(1, 2), (2, 1)], "unrolled")}
# The objective rule's basic runs: on T-B's releases (decides) and T-A's (information).
BASIC = {"T-B": (5, [(1, 2), (2, 1)]), "T-A": (1, [(1, 2)])}


def tuning_pairs():
    """Pair i = tuning images (2i, 2i + 1), i = 0..11, on checkpoint i // 4."""
    tuning = list(splits.JOURNAL.tuning)
    return [{"release": i, "image_indices": tuning[2 * i:2 * i + 2], "checkpoint": i // 4}
            for i in range(NUM_PAIRS)]


def pick_tv(mean_by_tv):
    """Highest mean SSIM; an exact tie goes to the smaller TV."""
    return max(sorted(mean_by_tv), key=lambda tv: (mean_by_tv[tv], -tv))


def edge_of(tv, grid):
    if tv == min(grid):
        return "low"
    if tv == max(grid):
        return "high"
    return None


def main_objective(basic_mean, cosine_mean, margin=OBJECTIVE_MARGIN):
    """Cosine unless basic is higher by `margin` or more."""
    return "basic" if basic_mean - cosine_mean >= margin - 1e-12 else "cosine"


def make_task(args, cell, pair, split, attacker, tv, e):
    seeds = [90000 + 10 * pair["release"] + j for j in range(args.restarts)]
    tag = "basic" if attacker == "lbfgs" else f"tv{tv:g}"
    path = os.path.join(args.out, args.setting, cell, f"{tag}_{split[0]}x{split[1]}_p{pair['release']:02d}.json")
    return {"setting": args.setting, "n": 2, "e": e, "split": list(split), "attacker": attacker, **pair,
            "local_lr": args.local_lr, "tv_key": cell if attacker != "lbfgs" else None,
            "tv": tv if attacker != "lbfgs" else None, "restart_seeds": seeds, "iters": args.iters,
            "cosine_iters": args.cosine_iters, "unrolled_iters": args.unrolled_iters,
            "releases": "tv-tuning", "git_commit": args.git_commit, "path": path}


def tv_tasks(args, cell, tvs):
    e, split_list, attacker = CELLS[cell]
    return [make_task(args, cell, pair, split, attacker, tv, e)
            for tv in tvs for pair in tuning_pairs() for split in split_list]


def basic_tasks(args):
    return [make_task(args, f"basic-{cell}", pair, split, "lbfgs", None, e)
            for cell, (e, split_list) in BASIC.items() for pair in tuning_pairs() for split in split_list]


def run(tasks, workers, label):
    """Run the tasks not yet done (longest first); return every task's record."""
    for t in tasks:
        if os.path.exists(t["path"]):
            am.check_finished(t)
    todo = [t for t in tasks if not os.path.exists(t["path"])]
    cost = {"unrolled": 3, "lbfgs": 2, "cosine": 1}
    todo.sort(key=lambda t: -cost[t["attacker"]])
    print(f"{label}: {len(tasks)} tasks, {len(tasks) - len(todo)} done, {len(todo)} to run", flush=True)
    start = time.time()
    if todo:
        with multiprocessing.Pool(workers, initializer=am._init_worker, initargs=(1,)) as pool:
            for k, rec in enumerate(pool.imap_unordered(am.safe_run_task, todo, chunksize=1)):
                if "fatal" in rec or "error" in rec:
                    pool.terminate()
                    sys.exit(f"STOP: {rec.get('fatal') or rec.get('error')} ({rec['path']})")
                if (k + 1) % 10 == 0 or k + 1 == len(todo):
                    print(f"  {k + 1}/{len(todo)} done, wall {time.time() - start:.0f}s", flush=True)
    out = []
    for t in tasks:
        with open(t["path"]) as f:
            out.append(json.load(f))
    return out


def mean_ssim(records):
    return float(np.mean([np.mean(r["ssim"]) for r in records]))


def tune_cell(args, cell):
    """TV grid for one tuning cell, with the edge rule."""
    grid = list(TVS)
    extended = []
    while True:
        recs = run(tv_tasks(args, cell, grid), args.workers, f"{args.setting} {cell} grid {grid}")
        by_tv = {tv: [r for r in recs if r["tv"] == tv] for tv in grid}
        means = {tv: mean_ssim(rs) for tv, rs in by_tv.items()}
        best = pick_tv(means)
        edge = edge_of(best, grid)
        ext = EDGE_EXTENSION.get(best)
        if edge and ext is not None and ext not in grid:
            extended.append(ext)
            grid = sorted(grid + [ext])
            continue
        break
    split_means = {tv: {f"{s[0]}x{s[1]}": mean_ssim([r for r in by_tv[tv] if tuple(r["split"]) == s])
                        for s in CELLS[cell][1]} for tv in grid}
    return {"chosen": best, "at_edge": edge, "extended_with": extended, "grid": grid,
            "mean_ssim_by_tv": {f"{tv:g}": means[tv] for tv in grid},
            "mean_ssim_by_tv_and_split": {f"{tv:g}": v for tv, v in split_means.items()},
            "releases": len(tuning_pairs()), "records": len(recs),
            "seconds_total": float(sum(r["seconds"] for r in recs))}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--setting", choices=["mnist-fresh", "mnist-trained"], required=True)
    p.add_argument("--cells", nargs="+", choices=list(CELLS), default=list(CELLS))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--cosine-iters", type=int, default=3000)
    p.add_argument("--unrolled-iters", type=int, default=3000)
    p.add_argument("--local-lr", default="auto", help="a number, or auto: the learning-rate rule pick")
    p.add_argument("--lr-rule", default=am.LR_RULE)
    p.add_argument("--out", default=os.path.join(am.PILOT_DIR, "tv"))
    p.add_argument("--summary-dir", default=am.PILOT_DIR)
    args = p.parse_args()
    args.git_commit = am.git_commit()
    args.local_lr = am.local_lr_for(args, args.setting)
    start = time.time()

    result = {"setting": args.setting, "local_lr": args.local_lr, "tv_grid": TVS,
              "rule": "highest mean matched SSIM, exact tie -> smaller TV; edge rule once per side "
                      "", "git_commit": args.git_commit, "cells": {}}
    # First pass: every cell's grid (and the basic runs) in one pool, longest tasks first.
    first = [t for cell in args.cells for t in tv_tasks(args, cell, TVS)]
    if args.setting == "mnist-fresh" and "T-B" in args.cells:
        first += basic_tasks(args)
    run(first, args.workers, f"{args.setting} first pass")
    for cell in args.cells:
        result["cells"][cell] = tune_cell(args, cell)
        c = result["cells"][cell]
        print(f"{args.setting} {cell}: " + ", ".join(f"TV {k} -> {v:.4f}" for k, v in c["mean_ssim_by_tv"].items())
              + f"; chosen {c['chosen']:g}" + (f" (still at the {c['at_edge']} edge)" if c["at_edge"] else ""),
              flush=True)
    result["chosen"] = {cell: c["chosen"] for cell, c in result["cells"].items()}

    if args.setting == "mnist-fresh" and "T-B" in result["cells"]:
        recs = run(basic_tasks(args), args.workers, "objective rule: basic L-BFGS")
        tb = result["cells"]["T-B"]
        cosine = tb["mean_ssim_by_tv"][f"{tb['chosen']:g}"]
        basic_b = mean_ssim([r for r in recs if r["e"] == 5])
        basic_a = mean_ssim([r for r in recs if r["e"] == 1])
        obj = {"rule": "cosine is the main objective unless basic is higher by 0.02 or more "
                       "",
               "basic_mean_ssim_T-B": basic_b, "cosine_mean_ssim_T-B": cosine, "cosine_tv_T-B": tb["chosen"],
               "basic_by_split_T-B": {f"{s[0]}x{s[1]}": mean_ssim([r for r in recs if r["e"] == 5
                                                                    and tuple(r["split"]) == s])
                                      for s in BASIC["T-B"][1]},
               "info_basic_mean_ssim_T-A": basic_a,
               "info_cosine_mean_ssim_T-A": result["cells"].get("T-A", {}).get("mean_ssim_by_tv", {}).get(
                   f"{result['chosen'].get('T-A', 0):g}"),
               "main_objective": main_objective(basic_b, cosine)}
        result["objective"] = obj
        with open(os.path.join(args.summary_dir, "objective.json"), "w") as f:
            json.dump(obj, f, indent=1)
        print(f"objective rule: basic {basic_b:.4f} vs cosine {cosine:.4f} -> {obj['main_objective']}", flush=True)

    result["wall_seconds"] = time.time() - start
    path = am.TV_FILE.format(setting=args.setting) if args.summary_dir == am.PILOT_DIR else \
        os.path.join(args.summary_dir, f"tv_{args.setting}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
