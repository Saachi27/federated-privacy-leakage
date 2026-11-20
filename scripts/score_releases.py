"""EXPLORATORY: risk per release on the saved study reconstructions."""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import recon_store, release_risk as rr, stats
from src.data import load_datasets

SETTINGS = ["mnist-fresh", "mnist-trained", "cifar10-fresh", "mnist-fedavg"]
TITLES = {"mnist-fresh": "MNIST, fresh LeNet, one FedSGD step",
          "mnist-trained": "MNIST, trained LeNet, one FedSGD step",
          "cifar10-fresh": "CIFAR-10, fresh LeNet, one FedSGD step",
          "mnist-fedavg": "MNIST, fresh LeNet, FedAvg update (5 local steps, one client)"}
MEASURES = [("mean_ssim", "Mean SSIM"), ("any_pass", "Any image passes (SSIM ≥ 0.9)"),
            ("hit", "Retrieval hit (top-1 in class)")]


def load_runs(setting_dir):
    """No-noise records with their reconstruction arrays, and the run arguments."""
    out, args = [], None
    for prefix, sigma_key in (("noise", True), ("sum16", False)):
        for path in sorted(glob.glob(os.path.join(setting_dir, f"{prefix}_s[0-9]*.json"))):
            with open(path) as f:
                data = json.load(f)
            args = args or data["args"]
            recons = recon_store.load(path.replace(".json", "_recon.npz"))
            for r in data["records"]:
                if sigma_key and r["sigma"] != 0:
                    continue
                out.append({**r, "recon": recons[r["recon_id"]]})
    return out, args


def as_nchw(data):
    t = torch.as_tensor(np.asarray(data)).float() / 255.0
    return t.unsqueeze(1) if t.dim() == 3 else t.permute(0, 3, 1, 2)


def chance_hit(release_labels, gallery_sizes):
    """Probability that K uniform random picks (one per target, in its class) hit the release."""
    counts = {c: release_labels.count(c) for c in set(release_labels)}
    miss = 1.0
    for c in release_labels:
        miss *= 1.0 - counts[c] / gallery_sizes[c]
    return 1.0 - miss


class Data:
    def __init__(self, dataset):
        _, test_ds, _ = load_datasets(dataset, normalize=False)
        self.images = as_nchw(test_ds.data)
        self.labels = np.array([int(y) for y in test_ds.targets])
        assert torch.allclose(self.images[50], test_ds[50][0])
        self.gallery = rr.Gallery(self.images, self.labels, np.arange(len(self.labels)))
        self.guesses = np.load(f"results/study/baseline_guesses_{dataset}.npz")
        self.class_average = torch.as_tensor(self.guesses["class_average"]).float()


def release_ids(seed, k, group, first, num):
    order = np.random.default_rng(seed).permutation(num)
    return [first + int(i) for i in order[group * k:(group + 1) * k]]


def score_setting(setting, data, root):
    records, args = load_runs(os.path.join(root, setting))
    first, num = args["first_image"], args["num_images"]
    rows, max_dev = [], 0.0
    base_cache = {}
    for r in records:
        k, seed, group = r["k"], r["seed"], r["group"]
        ids = release_ids(seed, k, group, first, num)
        targets = data.images[ids]
        labels = [int(data.labels[i]) for i in ids]
        # Saved reconstructions are already matched: recon[j] belongs to targets[j].
        att = rr.score_release(targets, r["recon"].astype(np.float32), labels, ids, data.gallery, matched=True)
        max_dev = max(max_dev, float(np.max(np.abs(np.array(att["ssims"]) - np.array(r["ssim"])))))
        key = (seed, k, group)
        if key not in base_cache:
            gk = f"k{k}_s{seed}_g{group}"
            assert list(data.guesses[f"target_ids_{gk}"]) == ids, f"grouping differs for {setting} {gk}"
            same = rr.score_release(targets, data.guesses[f"same_{gk}"], labels, ids, data.gallery,
                                    recon_classes=labels)
            avg = rr.score_release(targets, data.class_average[labels], labels, ids, data.gallery,
                                   recon_classes=labels)
            base_cache[key] = {
                **{m: max(same[m], avg[m]) for m, _ in MEASURES},
                "same": same, "avg": avg,
                "chance_hit": chance_hit(labels, data.gallery.sizes()),
            }
        own = float(np.mean([pick == i for pick, i in zip(att["picks"], ids)]))
        rows.append({"k": k, "attacker": r["attacker"], "seed": seed, "group": group,
                     "attack": att, "base": base_cache[key], "own_target": own})
    return rows, args, max_dev


def interval(a, b, strata, method="t"):
    return stats.paired_diff_interval(a, b, strata, level=0.95, method=method, n_resamples=10_000, seed=0)


def summarize(rows, ks, attackers):
    out = []
    for k in ks:
        for attacker in attackers:
            sel = [r for r in rows if r["k"] == k and r["attacker"] == attacker]
            if not sel:
                continue
            strata = [r["seed"] for r in sel]
            cell = {"k": k, "attacker": attacker, "n": len(sel), "seeds": len(set(strata)),
                    "chance_hit": float(np.mean([r["base"]["chance_hit"] for r in sel])),
                    "same_hit": float(np.mean([r["base"]["same"]["hit"] for r in sel])),
                    "avg_hit": float(np.mean([r["base"]["avg"]["hit"] for r in sel])),
                    "own_target": float(np.mean([r["own_target"] for r in sel]))}
            for m, _ in MEASURES:
                a = np.array([r["attack"][m] for r in sel], float)
                b = np.array([r["base"][m] for r in sel], float)
                zero = np.zeros_like(a)
                cell[m] = {"attack": interval(a, zero, strata), "base": interval(b, zero, strata),
                           "diff": interval(a, b, strata), "diff_boot": interval(a, b, strata, "bootstrap")}
            out.append(cell)
    return out


def f_ci(c, signed=False):
    sign = "+" if signed else ""
    if not np.isfinite(c["lo"]):
        return f"{c['mean']:{sign}.3f}"
    return f"{c['mean']:{sign}.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"


def f_diff(c, boot):
    s = f"{f_ci(c, True)} (boot [{boot['lo']:.3f}, {boot['hi']:.3f}])"
    return f"**{s}**" if stats.beats_baseline(c["lo"]) else s


def report(results, sizes, elapsed):
    L = ["# Risk per Release on the Saved Study Reconstructions (EXPLORATORY)", "",
         "> **EXPLORATORY.** This reuses the old target images (MNIST/CIFAR-10 test 50–81), the old "
         "grouping and the old attack runs (seeds 0–2, no restarts). It is **not** the locked held-out "
         "design. Use it to see how the measures behave, not "
         "as a result. Made by `scripts/score_releases.py`.", "",
         "## How to read this", "",
         "- A **release** is one attacked upload: one group of K images (seed, K, group). No noise.",
         "- **Mean SSIM:** mean valid-window SSIM of the K matched pairs.",
         "- **Any image passes:** 1 if any matched reconstruction has SSIM ≥ 0.9. The table shows the rate.",
         "- **Retrieval hit:** for each reconstruction, rank *all* test images of its class by SSIM and take "
         "the top-1, without knowing which images are in the release. A hit if any pick is one of the "
         "release's images. The table shows the hit rate.",
         "- **Baseline (label-aware):** for the same release, the better (per measure) of two guesses that "
         "know the labels: the class-average training image, and a random other test image of the same "
         "class (best-matched). These are the guesses of `scripts/baselines.py`, regenerated with the same "
         "draws.",
         "- Every interval is a 95% interval over releases, **stratified by checkpoint** (seed). The main "
         "one is a stratified t-interval; the paired bootstrap (10,000 resamples, seed 0) is in brackets. "
         "**Bold** = the attack beats the baseline (lower end of the paired 95% t-interval of attack − "
         "baseline above 0).",
         "- Releases per cell: K = 1: 96, K = 2: 48, K = 4: 24, K = 8: 12, K = 16: 6 (two per seed, so "
         "its intervals are rough).", ""]
    for setting, (cells, max_dev) in results.items():
        L += [f"## {TITLES[setting]} (`{setting}`)", ""]
        for m, title in MEASURES:
            L += [f"**{title}**", "", "| K | Attacker | Releases | Attack | Baseline | Attack − baseline |",
                  "|---|---|---|---|---|---|"]
            for c in cells:
                d = c[m]
                L.append(f"| {c['k']} | {c['attacker']} | {c['n']} | {f_ci(d['attack'])} | {f_ci(d['base'])} "
                         f"| {f_diff(d['diff'], d['diff_boot'])} |")
            L.append("")
        L += ["**Share of single reconstructions whose top-1 pick is exactly their own target** "
              "(per image, attack only; shows that low-SSIM reconstructions can still identify the image):", "",
              "| Attacker | " + " | ".join(f"K={k}" for k in sorted({c['k'] for c in cells})) + " |",
              "|---|" + "---|" * len({c['k'] for c in cells})]
        for attacker in ("basic", "cosine"):
            L.append(f"| {attacker} | " + " | ".join(f"{c['own_target'] * 100:.0f}%" for c in cells
                                                    if c["attacker"] == attacker) + " |")
        L.append("")
        chance = {c["k"]: c["chance_hit"] for c in cells}
        L += ["Retrieval by chance (one uniform random pick per target in its class): " +
              ", ".join(f"K = {k}: {v * 100:.2f}%" for k, v in chance.items()) + ".",
              f"Check: recomputed SSIM of the saved reconstructions differs from the saved records by at most "
              f"{max_dev:.1e} (float16 storage).", ""]
    L += ["## Gallery sizes (all test images of each class)", "",
          "| Dataset | " + " | ".join(str(c) for c in range(10)) + " |", "|---|" + "---|" * 10]
    for d, s in sizes.items():
        L.append(f"| {d} | " + " | ".join(str(s[c]) for c in range(10)) + " |")
    L += ["", "## Limits of this exploratory pass", "",
          "- **Old images and grouping.** Targets are test 50–81, the same images as the conference study; "
          "tuning used 0–49. Not the locked held-out design.",
          "- **Reconstruction labels.** The attack scripts save reconstructions already matched to targets, "
          "but not the label each reconstruction was optimized for. Here each reconstruction takes the "
          "label of its matched target. This is exact for K = 1 and for groups with one class. For mixed "
          "groups, a failed reconstruction may be searched in its matched target's class instead of its "
          "own, which can only help the attacker slightly. New runs should save the attacker's label per "
          "reconstruction.",
          "- **Same-class guess retrieval is always 0.** That guess is itself a test image (index ≥ 82) and "
          "so is in the gallery; its top-1 is itself, never a release image. The class-average guess "
          "picks one fixed typical image per class. So the retrieval baseline here is almost always 0; "
          "the chance line above is the fairer reference. The locked design must choose guesses that are "
          "not gallery members, or exclude a guess's own image.",
          "- **Retrieval control.** Across the MNIST settings at K = 8 and 16, no top-1 pick was a study image "
          "(test 50–81) outside its own release, and only 1 of 200 random-noise queries landed in 50–81 "
          "(chance about 0.6). So hits are not a bias towards the study images (checked by hand on "
          "21 Nov 2026; not part of this script).",
          "- **Few releases at large K.** K = 8 has 12 releases and K = 16 has 6, over 3 checkpoints.",
          "- **Seeds are checkpoints.** For the fresh settings, a seed is a fresh model, so the strata are "
          "fresh initialisations.",
          "", f"Run time: {elapsed:.0f} s."]
    return L


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/study")
    p.add_argument("--settings", nargs="+", default=SETTINGS)
    p.add_argument("--out", default="results/study/release_risk_exploratory.md")
    args = p.parse_args()
    torch.set_num_threads(2)
    start = time.time()
    datas, results, sizes = {}, {}, {}
    for setting in args.settings:
        dataset = "cifar10" if setting.startswith("cifar10") else "mnist"
        if dataset not in datas:
            datas[dataset] = Data(dataset)
            sizes[dataset] = datas[dataset].gallery.sizes()
        rows, run_args, max_dev = score_setting(setting, datas[dataset], args.root)
        ks = sorted({r["k"] for r in rows})
        cells = summarize(rows, ks, ["basic", "cosine"])
        results[setting] = (cells, max_dev)
        print(f"{setting}: {len(rows)} releases scored ({time.time() - start:.0f}s)", flush=True)
    lines = report(results, sizes, time.time() - start)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
