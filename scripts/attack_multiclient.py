"""The matched multi-client test."""

import argparse
import glob
import json
import multiprocessing
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.attack_sum import match_with_order
from src import checkpoints, multiclient as mc, recon_store, splits
from src.data import load_datasets
from src.settings import SETTINGS, build_target

# Kind -> (attacker number, objective)
ATTACKER_INFO = {"lbfgs": (1, "basic"), "cosine": (1, "cosine"), "unrolled": (2, "cosine")}
ATTACKER_1 = ["lbfgs", "cosine"]
ATTACKERS = ATTACKER_1 + ["unrolled"]
CHECKPOINTS = [0, 1, 2]
LR_GRID = [0.01, 0.03, 0.1, 0.3]
DRIFT_TARGET = 0.9
E1_DRIFT_MIN = 0.9999  # Below this at E = 1 is a bug
# The cut order: each level drops one more (setting, N) block.
CUTS = [("mnist-trained", 4), ("mnist-fresh", 4), ("mnist-trained", 2)]
PILOT_DIR = "results/journal/pilot"
LR_RULE = os.path.join(PILOT_DIR, "lr_rule.json")
# The learning-rate rule's file per dataset; the MNIST file is never rewritten for another dataset.
LR_RULE_BY_DATASET = {"mnist": LR_RULE, "cifar10": os.path.join(PILOT_DIR, "lr_rule_cifar10.json")}
SETTING_CHOICES = ["mnist-fresh", "mnist-trained", "cifar10-fresh"]
TV_FILE = os.path.join(PILOT_DIR, "tv_{setting}.json")
OUT = {"targets": "results/journal/targets", "pilot": os.path.join(PILOT_DIR, "grid"),
       "range": "results/journal/smoke"}
# Fields a finished record must share with its task; anything else is a different run.
TASK_KEYS = ["setting", "n", "e", "split", "attacker", "release", "checkpoint", "image_indices",
             "local_lr", "tv", "restart_seeds", "iters", "cosine_iters", "unrolled_iters", "label_perm"]


class Fatal(RuntimeError):
    """An error that must stop the whole run (a checkpoint hash or E = 1 drift check failed)."""


def restart_seed(n, r):
    """Seed of restart 0 of release r; restart j uses this + j."""
    return 10000 * n + 10 * r


def dataset_of(setting):
    return SETTINGS[setting]["dataset"]


def target_releases(n, dataset="mnist"):
    """The 30 target releases for N, each on checkpoint r // 10."""
    rel = splits.load_split_file(dataset).releases(n)
    return [{"release": r, "image_indices": idx, "checkpoint": r // 10} for r, idx in enumerate(rel)]


def pilot_releases(n):
    """Tuning images only: group i = images n*i .."""
    tuning = list(splits.JOURNAL.tuning)
    return [{"release": i, "image_indices": tuning[n * i:n * (i + 1)], "checkpoint": i % 3}
            for i in range(len(tuning) // n)]


def range_releases(n, count, target_range, seed, dataset="mnist"):
    """Smoke tests only: disjoint groups drawn from a range that holds no target image."""
    pool = np.arange(*target_range)
    clash = set(pool.tolist()) & set(splits.journal_splits(dataset).targets)
    assert not clash, f"--target-range holds {len(clash)} target images; smoke tests may not use targets"
    assert count * n <= len(pool), f"{count} groups of {n} need more than {len(pool)} images"
    order = np.random.default_rng(seed).permutation(pool)
    return [{"release": g, "image_indices": [int(i) for i in order[g * n:(g + 1) * n]], "checkpoint": g % 3}
            for g in range(count)]


def releases_for(args, n, dataset="mnist"):
    if args.target_range is not None:
        rel = range_releases(n, args.first_group + args.groups, args.target_range, args.group_seed, dataset)
    elif args.releases == "pilot":
        rel = pilot_releases(n)  # tuning images 0-49: the same indices for every
    else:
        rel = target_releases(n, dataset)
    return rel[args.first_group:args.first_group + args.groups]


def cells(n, es, split_kinds, attackers):
    """(E, split, attacker) cells for N images."""
    out = []
    for e in es:
        if e == 1:
            out += [(1, (1, n), a) for a in attackers if a in ATTACKER_1]
            continue
        for kind in split_kinds:
            if kind == "pairs" and n < 4:
                continue  # with N = 2 a 2-image client is the 1 x N split
            split = {"single": (1, n), "multi": (n, 1), "pairs": (n // 2, 2)}[kind]
            out += [(e, split, a) for a in attackers]
    return out


def tv_key(e, attacker):
    """Which tuned TV a cell uses, or None for the basic objective."""
    if ATTACKER_INFO[attacker][1] == "basic":
        return None
    if e == 1:
        return "T-A"
    return "T-B" if attacker == "cosine" else "T-C"


def task_path(out, setting, n, e, split, attacker, release):
    return os.path.join(out, setting, f"N{n}", f"E{e}_{split[0]}x{split[1]}_{attacker}_r{release:02d}.json")


def blocks(args):
    """(setting, N) blocks to run, in priority order (the reverse of the cut order)."""
    cut = set(CUTS[:args.cut])
    return [(s, n) for n in sorted(args.ns) for s in args.settings if (s, n) not in cut]


def make_tasks(args):
    tasks = []
    for setting, n in blocks(args):
        lr = local_lr_for(args, setting)
        for rel in releases_for(args, n, dataset_of(setting)):
            seed = restart_seed(n, rel["release"])
            for e, split, attacker in cells(n, args.es, args.splits, args.attackers):
                key = tv_key(e, attacker)
                tasks.append({
                    "setting": setting, "n": n, "e": e, "split": list(split), "attacker": attacker,
                    **rel, "local_lr": lr, "tv_key": key,
                    "tv": None if key is None else cosine_tv_for(args, setting, key),
                    "restart_seeds": [seed + j for j in range(args.restarts)],
                    "iters": args.iters, "cosine_iters": args.cosine_iters,
                    "unrolled_iters": args.unrolled_iters, "releases": args.releases_label,
                    "label_perm": args.label_perm,
                    "git_commit": args.git_commit,
                    "path": task_path(args.out, setting, n, e, split, attacker, rel["release"])})
    return tasks


def cosine_tv_for(args, setting, key):
    """--cosine-tv, else the pilot's tuned TV for this setting and tuning cell."""
    if args.cosine_tv is not None:
        return args.cosine_tv
    path = TV_FILE.format(setting=setting)
    if not os.path.exists(path):
        sys.exit(f"{path} not found: run scripts/tune_journal.py first, or pass --cosine-tv")
    with open(path) as f:
        return float(json.load(f)["chosen"][key])


def lr_rule_path(args, setting):
    """--lr-rule if given, else the learning-rate rule file of the setting's dataset."""
    return args.lr_rule or LR_RULE_BY_DATASET[dataset_of(setting)]


def local_lr_for(args, setting):
    if args.local_lr != "auto":
        return float(args.local_lr)
    path = lr_rule_path(args, setting)
    if not os.path.exists(path):
        sys.exit(f"{path} not found: run --lr-grid first, or pass --local-lr")
    with open(path) as f:
        return float(json.load(f)["settings"][setting]["chosen_lr"])


_IMAGES = {}
_HASHES = {}


def _dataset(dataset):
    if dataset not in _IMAGES:
        _, test_ds, meta = load_datasets(dataset, normalize=False)
        _IMAGES[dataset] = (test_ds, meta)
    return _IMAGES[dataset]


def load_images(dataset, indices):
    """(image, label) for these test indices, cached per process."""
    test_ds, meta = _dataset(dataset)
    pairs = [test_ds[i] for i in indices]
    return torch.stack([p[0] for p in pairs]), torch.tensor([int(p[1]) for p in pairs]), meta


def load_checkpoint(setting_name, meta, k):
    """The model of checkpoint k and its verified hash (a mismatch is Fatal)."""
    setting = SETTINGS[setting_name]
    model = build_target(meta, k, setting["model_path"])
    try:
        h = checkpoints.verify(model, k, setting["model_path"], dataset=setting["dataset"])
    except RuntimeError as err:
        raise Fatal(f"{setting_name}: {err}") from err
    return model, h


def current_checkpoint_hash(setting_name, k):
    """The hash of checkpoint k as it is now (verified against the pinned hash), cached."""
    if (setting_name, k) not in _HASHES:
        _, meta = _dataset(SETTINGS[setting_name]["dataset"])
        _HASHES[(setting_name, k)] = load_checkpoint(setting_name, meta, k)[1]
    return _HASHES[(setting_name, k)]


def write_atomic(path, write):
    """Call write(file) on a temporary file, then rename it to path."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def _init_worker(threads):
    torch.set_num_threads(threads)
    warnings.filterwarnings("ignore")


def attack(model, observed, x, y, task, meta):
    """Run the task's attacker with its restarts (handled in src/multiclient.py)."""
    split, e, lr = tuple(task["split"]), task["e"], task["local_lr"]
    seed, restarts = task["restart_seeds"][0], len(task["restart_seeds"])
    if task["attacker"] == "unrolled":
        return mc.reconstruct_unrolled(model, observed, x.shape, y, split, e, lr, iters=task["unrolled_iters"],
                                       tv_weight=task["tv"], seed=seed, restarts=restarts)
    return mc.attack_one(model, observed, x.shape, y, task["attacker"], meta["num_classes"],
                         iters=task["iters"], cosine_iters=task["cosine_iters"],
                         tv_weight=task["tv"] if task["tv"] is not None else 0.0, seed=seed, restarts=restarts)


def run_task(t):
    """Attack one release in one cell and save its record."""
    start = time.time()
    setting = SETTINGS[t["setting"]]
    x, y, meta = load_images(setting["dataset"], t["image_indices"])
    model, ckpt_hash = load_checkpoint(t["setting"], meta, t["checkpoint"])
    split, e, lr = tuple(t["split"]), t["e"], t["local_lr"]
    observed = mc.make_release(model, x, y, split, e, lr)  # always the true label order
    drift, norm_ratio = mc.drift(model, x, y, e, lr)
    if e == 1 and not drift >= E1_DRIFT_MIN:
        raise Fatal(f"E = 1 drift {drift} < {E1_DRIFT_MIN} on release {t['release']}: a bug")
    # The attacker may assign the labels to clients in another order.
    y_attacker = y[t["label_perm"]] if t.get("label_perm") is not None else y
    recon, info = attack(model, observed, x, y_attacker, t, meta)
    records, _, order = match_with_order(recon, list(x))
    slot_to_target = [order.index(i) for i in range(len(order))]
    attacker_number, objective = ATTACKER_INFO[t["attacker"]]
    batch = split[1]
    rec = {k: v for k, v in t.items() if k != "path"}
    rec.update({
        "attacker_number": attacker_number, "objective": objective,
        "split_name": f"{split[0]}x{split[1]}", "checkpoint_hash": ckpt_hash,
        "labels": y.tolist(),
        "client_partition": [t["image_indices"][c * batch:(c + 1) * batch] for c in range(split[0])],
        "drift_cosine": drift, "norm_ratio": norm_ratio, "release_norm": mc.norm(observed),
        "restart_losses": info["restart_losses"], "best_restart": info["best_restart"],
        "final_loss": info["final_loss"],
        "slot_labels": [int(v) for v in info["label"]],
        "slot_to_target": slot_to_target, "target_to_slot": order,
        "ssim": [r["ssim"] for r in records], "ssim_zeropad": [r["ssim_zeropad"] for r in records],
        "psnr": [r["psnr"] for r in records], "mse": [r["mse"] for r in records],
        "recon_order": "slot", "recon_file": os.path.basename(t["path"]).replace(".json", "_recon.npz"),
        "seconds": time.time() - start})
    os.makedirs(os.path.dirname(t["path"]), exist_ok=True)
    # The npz goes first; the JSON marks the task as finished.
    write_atomic(t["path"].replace(".json", "_recon.npz"),
                 lambda f: np.savez_compressed(f, recon_slots=recon_store.to_array(recon)))
    write_atomic(t["path"], lambda f: f.write(json.dumps(rec, indent=1).encode()))
    return rec


def safe_run_task(t):
    """Run_task, but an ordinary error is returned (and logged) instead of killing the pool."""
    try:
        return run_task(t)
    except Fatal as err:
        return {"fatal": str(err), "path": t["path"]}
    except Exception as err:  # a crashed task is rerun later with the same seeds
        return {"error": f"{type(err).__name__}: {err}", "path": t["path"]}


def check_finished(task):
    """A finished record must be the run this task describes; else stop."""
    with open(task["path"]) as f:
        rec = json.load(f)
    bad = [k for k in TASK_KEYS if rec.get(k) != task.get(k)]
    if bad:
        sys.exit(f"{task['path']} was made with different settings ({', '.join(bad)}); "
                 "use a new --out or remove it on purpose")
    if not os.path.exists(task["path"].replace(".json", "_recon.npz")):
        sys.exit(f"{task['path']} has no reconstruction file")
    # A finished task must have been made on the checkpoint we have now.
    try:
        now = current_checkpoint_hash(task["setting"], task["checkpoint"])
    except Fatal as err:
        sys.exit(f"STOP: {err}")
    if rec.get("checkpoint_hash") != now:
        sys.exit(f"STOP: {task['path']} was made on checkpoint hash {rec.get('checkpoint_hash')}, but "
                 f"{task['setting']} checkpoint {task['checkpoint']} now hashes to {now}; "
                 "use a new --out or remove it on purpose")


def run_grid(args):
    tasks = make_tasks(args)
    done = [t for t in tasks if os.path.exists(t["path"])]
    for t in done:
        check_finished(t)
    todo = [t for t in tasks if not os.path.exists(t["path"])]
    print(f"{len(tasks)} tasks, {len(done)} already done, {len(todo)} to run", flush=True)
    if args.max_tasks is not None:
        todo = todo[:args.max_tasks]
    if args.dry_run:
        for t in todo:
            print(t["path"])
        return
    os.makedirs(args.out, exist_ok=True)
    log = os.path.join(args.out, "failures.log")
    start, failed = time.time(), 0
    with multiprocessing.Pool(args.workers, initializer=_init_worker, initargs=(args.threads,)) as pool:
        for k, rec in enumerate(pool.imap_unordered(safe_run_task, todo, chunksize=1)):
            if "fatal" in rec:
                pool.terminate()
                sys.exit(f"STOP: {rec['fatal']} ({rec['path']})")
            if "error" in rec:
                failed += 1
                with open(log, "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {rec['path']}: {rec['error']}\n")
                print(f"{k + 1}/{len(todo)} FAILED {rec['path']}: {rec['error']}", flush=True)
                continue
            print(f"{k + 1}/{len(todo)} {rec['setting']} N={rec['n']} E={rec['e']} {rec['split_name']} "
                  f"{rec['attacker']} r{rec['release']}: SSIM {np.mean(rec['ssim']):.3f}, "
                  f"drift {rec['drift_cosine']:.3f}, losses {['%.3g' % v for v in rec['restart_losses']]}, "
                  f"{rec['seconds']:.0f}s, wall {time.time() - start:.0f}s", flush=True)
    print(f"finished {len(todo) - failed} tasks in {time.time() - start:.0f}s"
          + (f"; {failed} failed (see {log}); run again to retry them" if failed else ""), flush=True)


def lr_tuning_pairs():
    """The 25 tuning pairs (2i, 2i + 1), i = 0..24."""
    tuning = list(splits.JOURNAL.tuning)
    return [tuning[2 * i:2 * i + 2] for i in range(len(tuning) // 2)]


def pick_lr(medians, target=DRIFT_TARGET):
    """Smallest lr with median drift <= target; else the largest lr (grid edge)."""
    passing = [lr for lr in sorted(medians) if medians[lr] <= target]
    return (passing[0], False) if passing else (max(medians), True)


def lr_grid(args):
    """The learning-rate rule per setting: median drift at N = 2, E = 5 over 25 tuning pairs x 3 checkpoints."""
    paths = {lr_rule_path(args, s) for s in args.settings}
    if len(paths) != 1:
        sys.exit("--lr-grid writes one file: run it for one dataset's settings at a time")
    path = paths.pop()
    if os.path.exists(path) and not args.force:
        sys.exit(f"{path} exists; pass --force to redo it")
    result = {"rule": f"smallest lr with median drift cosine <= {DRIFT_TARGET}, else the largest "
                      "", "n": 2, "local_steps": args.lr_grid_steps, "lrs": args.lrs,
              "pairs": lr_tuning_pairs(), "checkpoints": CHECKPOINTS, "git_commit": args.git_commit,
              "settings": {}}
    for setting in args.settings:
        start = time.time()
        drifts = {lr: [] for lr in args.lrs}
        hashes = {}
        for k in CHECKPOINTS:
            for idx in lr_tuning_pairs():
                x, y, meta = load_images(SETTINGS[setting]["dataset"], idx)
                model, hashes[k] = load_checkpoint(setting, meta, k)
                for lr in args.lrs:
                    drifts[lr].append(mc.drift_cosine(model, x, y, args.lr_grid_steps, lr))
        medians = {lr: float(np.median(v)) for lr, v in drifts.items()}
        chosen, edge = pick_lr(medians)
        result["settings"][setting] = {
            "medians": {str(k): v for k, v in medians.items()},
            "drifts": {str(k): v for k, v in drifts.items()}, "count_per_lr": len(drifts[chosen]),
            "chosen_lr": chosen, "at_grid_edge": edge, "checkpoint_hashes": hashes,
            "seconds": time.time() - start}
        print(f"{setting}: median drift " + ", ".join(f"lr {k:g} -> {v:.4f}" for k, v in medians.items()) +
              f"; chosen lr {chosen:g}" + (" (grid edge)" if edge else ""), flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_atomic(path, lambda f: f.write(json.dumps(result, indent=1).encode()))
    print(f"Saved -> {path}")


def summary(args):
    """Progress per block and a table of finished cells."""
    tasks = make_tasks(args)
    by_block = {}
    for t in tasks:
        b = by_block.setdefault((t["setting"], t["n"]), [0, 0])
        b[0] += 1
        b[1] += os.path.exists(t["path"])
    lines = ["| Setting | N | Tasks done | Of |", "|---|---|---|---|"]
    lines += [f"| {s} | {n} | {d} | {total} |" for (s, n), (total, d) in by_block.items()]
    log = os.path.join(args.out, "failures.log")
    if os.path.exists(log):
        with open(log) as f:
            lines += ["", f"Failures logged: {sum(1 for _ in f)} (see {log})"]
    rows = {}
    for path in sorted(glob.glob(os.path.join(args.out, "*", "N*", "E*_r*.json"))):
        with open(path) as f:
            r = json.load(f)
        key = (r["setting"], r["n"], r["e"], r["split_name"], r["attacker"])
        rows.setdefault(key, []).append(r)
    lines += ["", "| Setting | N | E | Split | Attacker | Releases | Mean SSIM | Any image SSIM≥0.9 "
              "| Median drift | Mean s/release |", "|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(rows):
        rs = rows[key]
        ssim = np.mean([np.mean(r["ssim"]) for r in rs])
        anyp = np.mean([max(r["ssim"]) >= 0.9 for r in rs])
        lines.append(f"| {' | '.join(map(str, key))} | {len(rs)} | {ssim:.3f} | {anyp * 100:.0f}% | "
                     f"{np.median([r['drift_cosine'] for r in rs]):.3f} | "
                     f"{np.mean([r['seconds'] for r in rs]):.0f} |")
    print("\n".join(lines))


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--settings", nargs="+", choices=SETTING_CHOICES,
                   default=["mnist-fresh", "mnist-trained"])
    p.add_argument("--ns", nargs="+", type=int, default=[2, 4], choices=[2, 4])
    p.add_argument("--es", nargs="+", type=int, default=[1, 5])
    p.add_argument("--splits", nargs="+", choices=["single", "multi", "pairs"], default=["single", "multi"],
                   help="single = 1 x N, multi = N x 1, pairs = N/2 x 2 (N >= 4; plan 21.A) (E > 1 only)")
    p.add_argument("--attackers", nargs="+", choices=ATTACKERS, default=ATTACKERS)
    p.add_argument("--cut", type=int, default=0, choices=range(len(CUTS) + 1),
                   help="apply the cut order up to this level (1: drop trained N=4, "
                        "2: also fresh N=4, 3: also trained N=2)")
    p.add_argument("--releases", choices=["targets", "pilot"], default="targets",
                   help="targets = the target releases; pilot = tuning images only")
    p.add_argument("--groups", type=int, default=None, help="releases per cell (default: all)")
    p.add_argument("--first-group", type=int, default=0)
    p.add_argument("--target-range", nargs=2, type=int, default=None, metavar=("START", "END"),
                   help="smoke tests only: draw releases from this range (no target images allowed)")
    p.add_argument("--group-seed", type=int, default=0, help="seed of the --target-range draw")
    p.add_argument("--local-lr", default="auto", help="a number, or auto: the learning-rate rule pick")
    p.add_argument("--lr-rule", default=None, help="default: the learning-rate rule file of each setting's dataset")
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--label-perm", nargs="+", type=int, default=None,
                   help="Attacker 2 gives slot i the label of image perm[i]; "
                        "only with --splits multi --attackers unrolled --es 5 and one N")
    p.add_argument("--iters", type=int, default=300, help="L-BFGS outer steps")
    p.add_argument("--cosine-iters", type=int, default=3000)
    p.add_argument("--unrolled-iters", type=int, default=3000)
    p.add_argument("--cosine-tv", type=float, default=None,
                   help="TV weight for every cosine cell (default: the pilot's tuned values)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    p.add_argument("--max-tasks", type=int, default=None, help="stop after this many new tasks")
    p.add_argument("--dry-run", action="store_true", help="list the tasks that would run")
    p.add_argument("--lr-grid", action="store_true", help="run the learning-rate rule instead of the grid")
    p.add_argument("--lrs", nargs="+", type=float, default=LR_GRID)
    p.add_argument("--lr-grid-steps", type=int, default=5)
    p.add_argument("--force", action="store_true", help="redo --lr-grid even if saved")
    p.add_argument("--summary", action="store_true", help="progress and a table of finished tasks")
    p.add_argument("--out", default=None, help="default: results/journal/targets (or pilot/grid, smoke)")
    args = p.parse_args(argv)
    if args.target_range is not None:
        args.target_range = tuple(args.target_range)
        args.releases_label = f"range {args.target_range[0]}-{args.target_range[1] - 1} seed {args.group_seed}"
        mode = "range"
    else:
        args.releases_label = args.releases
        mode = args.releases
    if args.groups is None:
        args.groups = 30 if mode != "pilot" else 10 ** 6
    if args.label_perm is not None:
        ok = (args.splits == ["multi"] and args.attackers == ["unrolled"] and 1 not in args.es
              and len(args.ns) == 1 and sorted(args.label_perm) == list(range(args.ns[0])))
        if not ok or args.out is None:
            sys.exit("--label-perm needs --splits multi --attackers unrolled, no E = 1, one N, "
                     "a permutation of 0..N-1 and its own --out")
    args.out = args.out or OUT[mode]
    args.git_commit = git_commit()
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    if args.lr_grid:
        lr_grid(args)
    elif args.summary:
        summary(args)
    else:
        run_grid(args)


if __name__ == "__main__":
    main()
