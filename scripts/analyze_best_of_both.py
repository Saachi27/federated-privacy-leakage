"""Post hoc check: the split gap when each release keeps its best attacker."""
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import stats  # noqa: E402

ROOT = "results/journal/targets"
OUT = "results/journal/best_of_both"
MARGIN = 0.05
SETS = {"attacker 2": ["unrolled"], "best of attacker 1 (cosine) and 2": ["cosine", "unrolled"]}


def load(root=ROOT):
    cells = collections.defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(root, "*", "N*", "E5_*.json"))):
        with open(path) as f:
            r = json.load(f)
        cells[(r["setting"], r["n"], r["split_name"], r["attacker"])][r["release"]] = (
            float(np.mean(r["ssim"])), r["checkpoint"])
    return cells


def best(cells, setting, n, split, kinds):
    out = {}
    for kind in kinds:
        for rel, (s, ck) in cells[(setting, n, split, kind)].items():
            if rel not in out or s > out[rel][0]:
                out[rel] = (s, ck)
    return out


def analyze(cells):
    rows = []
    for setting in ("mnist-fresh", "mnist-trained"):
        for n in (2, 4):
            for name, kinds in SETS.items():
                multi = best(cells, setting, n, f"{n}x1", kinds)
                single = best(cells, setting, n, f"1x{n}", kinds)
                rels = sorted(set(multi) & set(single))
                assert len(rels) == 30, (setting, n, name, len(rels))
                for r in rels:
                    assert multi[r][1] == single[r][1], "checkpoint mismatch"
                a = [multi[r][0] for r in rels]
                b = [single[r][0] for r in rels]
                strata = [multi[r][1] for r in rels]
                ci = stats.paired_diff_interval(a, b, strata, level=0.90, method="t")
                per_ck = {}
                for k in sorted(set(strata)):
                    idx = [i for i, s in enumerate(strata) if s == k]
                    per_ck[str(k)] = float(np.mean([a[i] - b[i] for i in idx]))
                rows.append({"setting": setting, "n": n, "attackers": name,
                             "mean_multi": float(np.mean(a)), "mean_single": float(np.mean(b)),
                             "gap": ci["mean"], "lo90": ci["lo"], "hi90": ci["hi"],
                             "lower_bound_above_margin": bool(ci["lo"] > MARGIN),
                             "gap_per_checkpoint": per_ck})
    return rows


def render(rows):
    lines = ["# Best-of-both-attackers check (post hoc, secondary)", "",
             "Per release, the higher mean SSIM of attacker 1 (cosine) and attacker 2, per split.",
             "Gap = N x 1 - 1 x N, 90% checkpoint-stratified paired t-interval, 30 releases on 3 checkpoints.",
             "Choosing the better attacker needs the true images: an evaluator's upper envelope, applied to both splits.",
             "It does not change the locked outcome. Script: scripts/analyze_best_of_both.py.", "",
             "| Setting | N | Attackers | Mean N x 1 | Mean 1 x N | Gap (90%) | Lower bound > +0.05 | Gap per checkpoint k0 / k1 / k2 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ck = " / ".join(f"{v:.3f}" for v in r["gap_per_checkpoint"].values())
        lines.append(f"| {r['setting']} | {r['n']} | {r['attackers']} | {r['mean_multi']:.3f} | {r['mean_single']:.3f} | "
                     f"{r['gap']:.3f} [{r['lo90']:.3f}, {r['hi90']:.3f}] | {'yes' if r['lower_bound_above_margin'] else 'no'} | {ck} |")
    return "\n".join(lines) + "\n"


def main():
    rows = analyze(load())
    with open(OUT + ".json", "w") as f:
        json.dump({"margin": MARGIN, "rows": rows}, f, indent=2)
    with open(OUT + ".md", "w") as f:
        f.write(render(rows))
    print(render(rows))


if __name__ == "__main__":
    main()
