"""The add-on experiments, analysed once."""

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.analyze_journal as aj  # noqa: E402
from src import release_risk as rr, stats  # noqa: E402

MAIN_KIND = "cosine"  # The main objective everywhere
KINDS = ("unrolled", "cosine", "lbfgs")
DATASET = {"mnist-fresh": "mnist", "mnist-trained": "mnist", "cifar10-fresh": "cifar10"}


def comparisons():
    """(group, name, setting, N, PC key, '1 x N'-role key fn, 'N x 1'-role key fn)."""
    out = []
    for s in ("mnist-fresh", "mnist-trained"):
        pc = (s, 4, 1, "1x4", MAIN_KIND)
        out.append(("21.A", "G_a = 2×2 − 1×4", s, 4, pc,
                    lambda k, s=s: (s, 4, 5, "1x4", k), lambda k, s=s: (s, 4, 5, "2x2", k)))
        out.append(("21.A", "G_b = 4×1 − 2×2", s, 4, pc,
                    lambda k, s=s: (s, 4, 5, "2x2", k), lambda k, s=s: (s, 4, 5, "4x1", k)))
    s = "mnist-fresh"
    out.append(("21.B", "E = 10: 2×1 − 1×2", s, 2, (s, 2, 1, "1x2", MAIN_KIND),
                lambda k, s=s: (s, 2, 10, "1x2", k), lambda k, s=s: (s, 2, 10, "2x1", k)))
    s = "cifar10-fresh"
    out.append(("21.C", "CIFAR-10: 2×1 − 1×2", s, 2, (s, 2, 1, "1x2", MAIN_KIND),
                lambda k, s=s: (s, 2, 5, "1x2", k), lambda k, s=s: (s, 2, 5, "2x1", k)))
    return out


def label(cells, baselines, n, pc_key, single_key, multi_key):
    """The same outcome order for one comparison; same steps as aj.label_block."""
    missing = [k for k in (pc_key, single_key, multi_key) if k not in cells]
    if missing:
        return {"outcome": "not run", "missing": [list(k) for k in missing]}
    pc = aj.versus_baseline(cells[pc_key], baselines, n)
    single = aj.versus_baseline(cells[single_key], baselines, n)
    multi = aj.versus_baseline(cells[multi_key], baselines, n)
    g = aj.gap(cells[multi_key], cells[single_key])
    enough = all(p["status"] == "decides" for p in (pc, single, multi, g))
    outcome = (stats.classify_outcome(pc["t95"]["lo"], single["t95"]["lo"], multi["t95"]["lo"],
                                      g["t90"]["lo"], g["t90"]["hi"]) if enough else stats.DESCRIPTIVE)
    any_gap = aj.gap(cells[multi_key], cells[single_key], score="a")
    out = {"outcome": outcome, "pc": pc, "single": single, "multi": multi, "gap": g, "any_pass_gap": any_gap,
           "single_any": float(np.mean([v["a"] for v in cells[single_key].values()])),
           "multi_any": float(np.mean([v["a"] for v in cells[multi_key].values()]))}
    if all("h" in v for v in cells[single_key].values()):
        out["single_hit"] = float(np.mean([v["h"] for v in cells[single_key].values()]))
        out["multi_hit"] = float(np.mean([v["h"] for v in cells[multi_key].values()]))
    return out


def gallery_for(dataset):
    from src.data import load_datasets
    _, test_ds, _ = load_datasets(dataset, normalize=False)
    images = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    labels = np.array([int(y) for y in test_ds.targets])
    return rr.Gallery(images, labels, np.arange(len(labels)))


def build(records_dirs, retrieval):
    records = [r for d in records_dirs if os.path.isdir(d) for r in aj.load_records(d)]
    cells = {}
    for ds in sorted({DATASET[r["setting"]] for r in records}):
        gallery = gallery_for(ds) if retrieval else None
        part = [r for r in records if DATASET[r["setting"]] == ds]
        cells.update(aj.build_cells(part, gallery))
    aj.check_cells_paired(cells)
    return cells


def analyze(cells, baselines):
    rows = []
    for group, name, setting, n, pc_key, single_fn, multi_fn in comparisons():
        for kind in KINDS:
            res = label(cells, baselines[DATASET[setting]], n, pc_key, single_fn(kind), multi_fn(kind))
            rows.append({"group": group, "comparison": name, "setting": setting, "n": n, "attacker": kind,
                         "single_split": single_fn(kind)[3], "multi_split": multi_fn(kind)[3], **res})
    return rows


def _iv(part, lv):
    ci = part.get(f"t{lv}") or part.get(f"boot{lv}")
    if ci is None:
        return "–"
    tag = "" if part.get(f"t{lv}") else " (descriptive)"
    return f"{ci['mean']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]{tag}"


def render(rows):
    L = ["# Add-on experiments", "",
         "*Add-on, written after the main result. Secondary: none of this changes the main outcome. "
         "Made by `scripts/analyze_addons.py` with the locked rules of `scripts/analyze_journal.py` "
         "(same outcome order, stratified paired t-intervals, ±0.05 margin).*", "",
         "In each row the split with fewer images per client plays the \"N × 1\" role. "
         "PC = the E = 1 positive control of that setting and N.", "",
         "| Group | Comparison | Setting | Attacker | Outcome | PC s − b (95%) | More-per-client split s − b (95%) | "
         "Fewer-per-client split s − b (95%) | Gap (90%) | Any image ≥ 0.9 (more / fewer) | Retrieval hit (more / fewer) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        name = f"{r['comparison']}"
        att = aj.KIND_LABEL[r["attacker"]]
        if r["outcome"] == "not run":
            L.append(f"| {r['group']} | {name} | {r['setting']} | {att} | not run | – | – | – | – | – | – |")
            continue
        hit = (f"{r['single_hit']:.2f} / {r['multi_hit']:.2f}" if "single_hit" in r else "–")
        L.append(f"| {r['group']} | {name} | {r['setting']} | {att} | **{r['outcome']}** | {_iv(r['pc'], 95)} | "
                 f"{_iv(r['single'], 95)} | {_iv(r['multi'], 95)} | {_iv(r['gap'], 90)} | "
                 f"{r['single_any']:.2f} / {r['multi_any']:.2f} | {hit} |")
    L += ["", "Mean SSIM per split (the s in s − b) is the estimate's mean plus the baseline mean in the JSON.", ""]
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", nargs="+", default=["results/journal/targets", "results/journal/addons"])
    p.add_argument("--baselines-mnist", default="results/journal/baselines_mnist.json")
    p.add_argument("--baselines-cifar10", default="results/journal/baselines_cifar10.json")
    p.add_argument("--no-retrieval", action="store_true")
    p.add_argument("--out", default="results/journal/addons_report")
    args = p.parse_args()
    torch.set_num_threads(4)
    baselines = {"mnist": aj.load_baselines(args.baselines_mnist)[0], "cifar10": {}}
    if os.path.exists(args.baselines_cifar10):
        baselines["cifar10"] = aj.load_baselines(args.baselines_cifar10)[0]
    cells = build(args.records, not args.no_retrieval)
    rows = analyze(cells, baselines)
    with open(args.out + ".json", "w") as f:
        json.dump(aj._jsonable(rows), f, indent=1)
    with open(args.out + ".md", "w") as f:
        f.write(render(rows))
    for r in rows:
        if r["attacker"] == "unrolled":
            print(f"{r['group']} {r['comparison']} {r['setting']}: {r['outcome']}")
    print(f"Saved -> {args.out}.md, {args.out}.json")


if __name__ == "__main__":
    main()
