"""The follow-up checks, analysed once."""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.analyze_addons as aa  # noqa: E402
import scripts.analyze_journal as aj  # noqa: E402

SETTINGS = ("mnist-fresh", "mnist-trained")
RELEASES = 30
R8 = "unrolled_r8"  # attacker 2, 8 restarts (22.A)
NOMAP = "unrolled_nomap"  # attacker 2, label-to-client map unknown (22.B)
LABEL = {"unrolled": "attacker 2 (2 restarts, locked)", R8: "attacker 2, 8 restarts",
         NOMAP: "attacker 2, label map unknown"}


def _loss(rec):
    """A NaN or infinite final objective counts as +inf."""
    v = rec["final_loss"]
    return v if v == v and v != float("-inf") else float("inf")


def as_kind(records, kind):
    out = []
    for r in records:
        r = copy.deepcopy(r)
        r["attacker"] = kind
        out.append(r)
    return out


def label_unknown(main, swapped):
    """22.B: per release, the order with the lower final objective (a tie keeps the true order)."""
    true_by = {(r["setting"], r["release"]): r for r in main
               if r["n"] == 2 and r["e"] == 5 and r["split_name"] == "2x1" and r["attacker"] == "unrolled"}
    out, picks = [], []
    for s in swapped:
        t = true_by[(s["setting"], s["release"])]
        aj.check_pair(t, s, "22.B true vs swapped order")
        assert s["label_perm"] == [1, 0] and t["labels"] == s["labels"]
        keep_true = _loss(t) <= _loss(s)
        chosen = copy.deepcopy(t if keep_true else s)
        chosen["attacker"] = NOMAP
        chosen["kept_order"] = "true" if keep_true else "swapped"
        out.append(chosen)
        if t["labels"][0] != t["labels"][1]:
            picks.append({"setting": s["setting"], "release": s["release"], "true_kept": keep_true})
    return out, picks


def complete(cells, key):
    return key in cells and len(cells[key]) == RELEASES


def row(cells, baselines, setting, n, kind, single_kind=None):
    """Section 12 label for N x 1 (kind) vs 1 x N (single_kind, default the same kind)."""
    single_kind = single_kind or kind
    pc = (setting, n, 1, f"1x{n}", aa.MAIN_KIND)
    single, multi = (setting, n, 5, f"1x{n}", single_kind), (setting, n, 5, f"{n}x1", kind)
    missing = [k for k in (pc, single, multi) if not complete(cells, k)]
    base = {"setting": setting, "n": n, "attacker": kind, "single_attacker": single_kind}
    if missing:
        return {**base, "outcome": "not run",
                "missing": [f"{list(k)} ({len(cells.get(k, {}))}/{RELEASES})" for k in missing]}
    res = aa.label(cells, baselines, n, pc, single, multi)
    res["single_mean"] = float(np.mean([v["s"] for v in cells[single].values()]))
    res["multi_mean"] = float(np.mean([v["s"] for v in cells[multi].values()]))
    return {**base, **res}


def restart_change(cells, setting, n):
    """22.A, descriptive: paired s(8 restarts) - s(2 restarts) per split."""
    out = {}
    for split in (f"1x{n}", f"{n}x1"):
        k8, k2 = (setting, n, 5, split, R8), (setting, n, 5, split, "unrolled")
        if complete(cells, k8) and complete(cells, k2):
            out[split] = aj.gap(cells[k8], cells[k2])
    return out


def analyze(cells, baselines, picks):
    a_rows, changes = [], {}
    for n in (2, 4):
        for s in SETTINGS:
            a_rows.append(row(cells, baselines, s, n, "unrolled"))
            a_rows.append(row(cells, baselines, s, n, R8))
            changes[f"{s} N={n}"] = restart_change(cells, s, n)
    b_rows = [row(cells, baselines, s, 2, NOMAP, single_kind="unrolled") for s in SETTINGS]
    b_rows = [dict(r, picks=[p for p in picks if p["setting"] == r["setting"]]) for r in b_rows]
    return {"22.A": a_rows, "22.A_restart_change": changes, "22.B": b_rows}


def _iv(part, lv):
    return aa._iv(part, lv) if part else "–"


def _table(rows, extra=None):
    L = ["| Setting | N | Attacker | Outcome | PC s − b (95%) | 1 × N s − b (95%) | N × 1 s − b (95%) | "
         "Gap N × 1 − 1 × N (90%) | Mean SSIM 1 × N / N × 1 | Any ≥ 0.9 1 × N / N × 1 | Retrieval 1 × N / N × 1 |"
         + (f" {extra[0]} |" if extra else ""),
         "|---|---|---|---|---|---|---|---|---|---|---|" + ("---|" if extra else "")]
    for r in rows:
        att = LABEL[r["attacker"]]
        if r["outcome"] == "not run":
            L.append(f"| {r['setting']} | {r['n']} | {att} | not run | – | – | – | – | – | – | – |"
                     + (" – |" if extra else ""))
            continue
        hit = f"{r['single_hit']:.2f} / {r['multi_hit']:.2f}" if "single_hit" in r else "–"
        L.append(f"| {r['setting']} | {r['n']} | {att} | **{r['outcome']}** | {_iv(r['pc'], 95)} | "
                 f"{_iv(r['single'], 95)} | {_iv(r['multi'], 95)} | {_iv(r['gap'], 90)} | "
                 f"{r['single_mean']:.3f} / {r['multi_mean']:.3f} | {r['single_any']:.2f} / {r['multi_any']:.2f} | "
                 f"{hit} |" + (f" {extra[1](r)} |" if extra else ""))
    return L


def render(res):
    L = ["# Follow-up checks", "",
         "*Follow-up, written after the main result and the add-ons. Secondary: none of this changes the main "
         "outcome or any add-on label. Made by `scripts/analyze_followup.py` with the locked rules of "
         "`scripts/analyze_journal.py` (same outcome order, stratified paired t-intervals, ±0.05 margin). "
         "PC = the E = 1 positive control of the main run.*", "",
         "## 22.A Attacker 2 with 8 restarts (E = 5)", "",
         "Each block shows the locked 2-restart row and the 8-restart row. The 8 restarts include the main "
         "run's 2 starts.", ""]
    L += _table(res["22.A"])
    L += ["", "Paired change in mean SSIM from 2 to 8 restarts (8 − 2; 90% and 95% stratified t-intervals; "
          "descriptive):", "", "| Block | Split | Change (90%) | Change (95%) |", "|---|---|---|---|"]
    for block, parts in res["22.A_restart_change"].items():
        if not parts:
            L.append(f"| {block} | – | not run | – |")
        for split, g in parts.items():
            L.append(f"| {block} | {split.replace('x', ' × ')} | {_iv(g, 90)} | {_iv(g, 95)} |")
    L += ["", "## 22.B Attacker 2 without the label-to-client map (N = 2, E = 5)", "",
          "The N × 1 attacker tries both label orders and keeps the one with the lower final objective. "
          "The 1 × 2 side is the main run (it needs no map).", ""]

    def kept(r):
        p = r.get("picks", [])
        return f"{sum(x['true_kept'] for x in p)}/{len(p)}" if p else "–"
    L += _table(res["22.B"], extra=("True order kept (different labels)", kept))
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", default="results/journal/targets")
    p.add_argument("--restarts8", default="results/journal/followup/restarts8")
    p.add_argument("--labelperm", default="results/journal/followup/labelperm")
    p.add_argument("--baselines-mnist", default="results/journal/baselines_mnist.json")
    p.add_argument("--no-retrieval", action="store_true")
    p.add_argument("--out", default="results/journal/followup_report")
    args = p.parse_args()
    torch.set_num_threads(4)
    load = lambda d: aj.load_records(d) if os.path.isdir(d) else []  # noqa: E731
    main_recs = [r for r in load(args.targets) if r["setting"] in SETTINGS]
    r8 = as_kind(load(args.restarts8), R8)
    nomap, picks = label_unknown(main_recs, load(args.labelperm))
    gallery = None if args.no_retrieval else aa.gallery_for("mnist")
    cells = aj.build_cells(main_recs + r8 + nomap, gallery)
    baselines = aj.load_baselines(args.baselines_mnist)[0]
    res = analyze(cells, baselines, picks)
    with open(args.out + ".json", "w") as f:
        json.dump(aj._jsonable(res), f, indent=1)
    with open(args.out + ".md", "w") as f:
        f.write(render(res))
    for r in res["22.A"] + res["22.B"]:
        print(f"{r['setting']} N={r['n']} {LABEL[r['attacker']]}: {r['outcome']}")
    print(f"Saved -> {args.out}.md, {args.out}.json")


if __name__ == "__main__":
    main()
