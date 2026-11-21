"""No-information baselines: what a "reconstruction" scores without any leak."""

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.attack_sum import match_images
from src import metrics, release_risk as rr, splits
from src.data import load_datasets

METRICS = ("ssim", "ssim_zeropad", "psnr", "mse")


def mean_ci(per_seed):
    """Mean over seeds and the 95% CI half-width (t distribution)."""
    a = np.asarray(per_seed, dtype=float)
    if len(a) < 2 or np.allclose(a, a[0]):
        return float(a.mean()), float("nan")
    return float(a.mean()), float(stats.t.ppf(0.975, len(a) - 1) * stats.sem(a))


def score(fakes, targets):
    recs = [metrics.all_metrics(f, t) for f, t in zip(fakes, targets)]
    return {m: float(np.mean([r[m] for r in recs])) for m in METRICS}


N_DRAWS = 5


def same_class_seed(n, r):
    return 50000 + 1000 * n + r


def _score_summary(out):
    return {"ssims": out["ssims"], "mean_ssim": out["mean_ssim"], "any_pass": out["any_pass"],
            "hit": out.get("hit"), "picks": out.get("picks")}


def release_baseline(n, r, targets, labels, release_ids, class_avg, prior_by_class, image_of, gallery,
                     average=None, n_draws=N_DRAWS):
    """Baselines for one release."""
    labels = [int(y) for y in labels]
    ca = rr.score_release(targets, class_avg[labels], labels, release_ids, gallery, recon_classes=labels)
    rng = np.random.default_rng(same_class_seed(n, r))
    draws = []
    for _ in range(n_draws):
        ids = [int(rng.choice(prior_by_class[y])) for y in labels]
        guesses = torch.stack([image_of(i) for i in ids])
        out = rr.score_release(targets, guesses, labels, release_ids, gallery, recon_classes=labels,
                               guess_ids=ids)
        draws.append({"ids": ids, **_score_summary(out)})
    a_r = ca["mean_ssim"]
    s_r = float(np.mean([d["mean_ssim"] for d in draws]))
    same_pass = float(np.mean([d["any_pass"] for d in draws]))
    same_hit = float(np.mean([d["hit"] for d in draws]))
    rec = {"n": n, "release": r, "image_indices": [int(i) for i in release_ids], "labels": labels,
           "class_average": _score_summary(ca), "same_class_draws": draws,
           "A": a_r, "S": s_r, "b": max(a_r, s_r),
           "any_pass_class_average": ca["any_pass"], "any_pass_same_share": same_pass,
           "baseline_any_pass": max(float(ca["any_pass"]), same_pass),
           "hit_class_average": ca["hit"], "hit_same_share": same_hit,
           "hit_exact_chance": rr.chance_hit(labels, gallery.sizes()),
           "baseline_hit": max(float(ca["hit"]), same_hit), "seed": same_class_seed(n, r)}
    context = {"all black": torch.zeros_like(targets)}
    if average is not None:
        context["average digit"] = average.expand_as(targets)
    g = torch.Generator().manual_seed(60000 + 1000 * n + r)
    context["random noise"] = torch.rand(targets.shape, generator=g)
    rec["context_mean_ssim"] = {k: rr.score_release(targets, v, labels)["mean_ssim"] for k, v in context.items()}
    return rec


def journal_main(args):
    from scripts.attack_multiclient import pilot_releases
    split = splits.load_split_file(args.dataset)  # asserts the four lists are disjoint
    split_path = splits.SPLIT_FILE.format(dataset=args.dataset)
    train_ds, test_ds, _ = load_datasets(args.dataset, normalize=False)
    test_labels = np.array([int(y) for y in test_ds.targets])
    prior = sorted(split.prior)
    prior_by_class = splits.pool_by_class(test_labels, prior)
    forbidden = set(split.tuning) | set(split.retired) | set(split.targets)
    assert not forbidden & set(prior)
    train = torch.as_tensor(np.asarray(train_ds.data)).float() / 255.0
    train = train.unsqueeze(1) if train.dim() == 3 else train.permute(0, 3, 1, 2)
    train_labels = torch.as_tensor(np.array([int(y) for y in train_ds.targets]))
    class_avg = torch.stack([train[train_labels == c].mean(0) for c in range(10)])
    test_images = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    gallery = rr.Gallery(test_images, test_labels, np.arange(len(test_ds)))

    out = {"dataset": args.dataset, "split_file": split_path, "split_sha256": splits.file_sha256(split_path),
           "releases_from": args.releases, "n_draws": N_DRAWS, "gallery_sizes": gallery.sizes(),
           "rule": "label-aware baseline per release", "releases": {}}
    for n in args.ns:
        rels = split.releases(n) if args.releases == "targets" else \
            [p["image_indices"] for p in pilot_releases(n)]
        recs = []
        for r, idx in enumerate(rels):
            rec = release_baseline(n, r, test_images[idx], test_labels[idx].tolist(), idx, class_avg,
                                   prior_by_class, lambda i: test_images[i], gallery, train.mean(0))
            assert all(i in set(prior) for d in rec["same_class_draws"] for i in d["ids"])
            recs.append(rec)
        out["releases"][str(n)] = recs
        print(f"N = {n}: {len(recs)} releases, mean b {np.mean([x['b'] for x in recs]):.3f}, "
              f"mean baseline hit {np.mean([x['baseline_hit'] for x in recs]):.3f}", flush=True)
    path = args.out_json or ("results/journal/baselines_mnist.json" if args.releases == "targets"
                             else "results/journal/pilot/baselines_pilot.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"Saved -> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p.add_argument("--splits", choices=list(splits.SPLITS), default="legacy",
                   help="image ranges from src/splits.py (legacy = old results)")
    p.add_argument("--first-image", type=int, default=None,
                   help="first target image (default: start of the split's target range)")
    p.add_argument("--num-images", type=int, default=32)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--out", default="results/baselines")
    p.add_argument("--save-guesses", default=None, help="optional .npz for the label-aware guess images")
    p.add_argument("--journal", action="store_true", help="per-release journal baselines")
    p.add_argument("--releases", choices=["targets", "pilot"], default="targets", help="journal mode only")
    p.add_argument("--ns", nargs="+", type=int, default=[2, 4], help="journal mode only")
    p.add_argument("--out-json", default=None, help="journal mode output")
    args = p.parse_args()
    torch.set_num_threads(2)
    if args.journal:
        return journal_main(args)
    split = splits.get(args.splits)
    first = split.targets.start if args.first_image is None else args.first_image
    num = args.num_images

    train_ds, test_ds, _ = load_datasets(args.dataset, normalize=False)
    assert split.is_disjoint(len(test_ds)), split.overlaps(len(test_ds))
    assert split.targets.contains(first, num, len(test_ds)), \
        f"targets {first}-{first + num - 1} are outside the {split.name} target range {split.targets}"
    targets = [test_ds[i][0] for i in range(first, first + num)]
    labels = [int(test_ds[i][1]) for i in range(first, first + num)]
    test_labels = np.array([int(y) for y in test_ds.targets])
    train_labels = torch.as_tensor(np.array([int(y) for y in train_ds.targets]))
    pool_idx = split.prior.indices(len(test_ds))
    pool_by_class = splits.pool_by_class(test_labels, split.prior)
    train = torch.as_tensor(np.asarray(train_ds.data)).float() / 255.0
    train = train.unsqueeze(1) if train.dim() == 3 else train.permute(0, 3, 1, 2)  # (N, C, H, W)
    average = train.mean(0)
    class_avg = {c: train[train_labels == c].mean(0) for c in range(10)}

    rows = {}  # name -> {metric: [per-seed values]}
    guesses = {"class_average": torch.stack([class_avg[c] for c in range(10)]).numpy()}

    def add(name, seed_scores):
        rows[name] = {m: [s[m] for s in seed_scores] for m in METRICS}

    add("all black", [score([torch.zeros_like(t) for t in targets], targets)])
    add("average digit", [score([average for _ in targets], targets)])
    add("class average (knows labels)", [score([class_avg[y] for y in labels], targets)])
    noise = []
    for seed in args.seeds:
        g = torch.Generator().manual_seed(seed)
        noise.append(score([torch.rand(t.shape, generator=g) for t in targets], targets))
    add("random noise", noise)

    for k in args.ks:
        unrelated, same_class = [], []
        for seed in args.seeds:
            order = np.random.default_rng(seed).permutation(num)  # same grouping as the experiments
            rng = np.random.default_rng(1000 + seed)
            u_recs, s_recs = [], []
            for group in range(num // k):
                idx = [int(i) for i in order[group * k:(group + 1) * k]]
                chosen = [targets[i] for i in idx]
                others = torch.stack([test_ds[int(i)][0] for i in rng.choice(pool_idx, k, replace=False)])
                u_recs += match_images(others, chosen)
                same_ids = [int(rng.choice(pool_by_class[labels[i]])) for i in idx]
                same = torch.stack([test_ds[j][0] for j in same_ids])
                s_recs += match_images(same, chosen)
                key = f"k{k}_s{seed}_g{group}"
                guesses[f"same_{key}"] = same.numpy()
                guesses[f"same_ids_{key}"] = np.array(same_ids)
                guesses[f"target_ids_{key}"] = np.array([first + i for i in idx])
            unrelated.append({m: float(np.mean([r[m] for r in u_recs])) for m in METRICS})
            same_class.append({m: float(np.mean([r[m] for r in s_recs])) for m in METRICS})
        add(f"unrelated digits, best-matched, K={k}", unrelated)
        add(f"same-class digits, best-matched, K={k} (knows labels)", same_class)
        # The line a leak must clear: the better label-aware guess, per seed and metric.
        cls = {m: rows["class average (knows labels)"][m][0] for m in METRICS}
        best = []
        for s in same_class:
            pick = {m: max(s[m], cls[m]) for m in ("ssim", "ssim_zeropad", "psnr")}
            pick["mse"] = min(s["mse"], cls["mse"])
            best.append(pick)
        add(f"label-aware baseline, K={k}", best)

    summary = {}
    for name, vals in rows.items():
        summary[name] = {m: dict(zip(("mean", "ci95"), mean_ci(v))) for m, v in vals.items()}
        summary[name]["seeds"] = len(vals["ssim"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump({"dataset": args.dataset, "splits": split.name, "images": f"{args.dataset} test {first}-{first + num - 1}", "seeds": args.seeds, "ks": args.ks,
                   "baselines": summary, "per_seed": rows}, f, indent=2)

    def fmt(d):
        return f"{d['mean']:.3f}" if np.isnan(d["ci95"]) else f"{d['mean']:.3f} ± {d['ci95']:.3f}"

    lines = [f"# No-information baselines ({args.dataset} test images {first}-{first + num - 1})", "",
             "A reconstruction must score clearly above the **label-aware baseline** to show a leak, "
             "because the attackers are given the labels. SSIM is the standard form; the old "
             "zero-padded SSIM is shown for comparison. ± is the 95% CI over seeds.", "",
             "| Fake reconstruction | SSIM | Old SSIM (zero-padded) | PSNR (dB) | MSE |",
             "|---|---|---|---|---|"]
    for name, s in summary.items():
        lines.append(f"| {name} | {fmt(s['ssim'])} | {fmt(s['ssim_zeropad'])} | {fmt(s['psnr'])} "
                     f"| {fmt(s['mse'])} |")
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved -> {args.out}.json, {args.out}.md")
    if args.save_guesses:
        np.savez_compressed(args.save_guesses, **guesses)
        print(f"Saved guesses -> {args.save_guesses}")


if __name__ == "__main__":
    main()
