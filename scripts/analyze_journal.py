"""Every journal table from one command."""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import release_risk as rr, stats

CHECKPOINTS = [0, 1, 2]
KIND_LABEL = {"lbfgs": "attacker 1, basic", "cosine": "attacker 1, cosine", "unrolled": "attacker 2, cosine"}
MAIN = ("mnist-fresh", 2)
BOOT = {"n_resamples": 10_000, "seed": 12345}


def load_records(root):
    """Every finished record under root, with the path of its reconstruction file."""
    out = []
    for path in sorted(glob.glob(os.path.join(root, "*", "N*", "E*_r*.json"))):
        with open(path) as f:
            rec = json.load(f)
        rec["recon_path"] = os.path.join(os.path.dirname(path), rec["recon_file"])
        out.append(rec)
    return out


def load_baselines(path):
    """{(N, release): baseline record} from scripts/baselines.py --journal."""
    with open(path) as f:
        data = json.load(f)
    return {(int(n), rec["release"]): rec for n, recs in data["releases"].items() for rec in recs}, data


def main_objective_from(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)["main_objective"]
    return None


def release_scores(rec, gallery):
    """S_r, a_r and h_r for one record."""
    ssim = [float(v) for v in rec["ssim"]]
    out = {"release": rec["release"], "checkpoint": rec["checkpoint"], "s": float(np.mean(ssim)),
           "a": int(max(ssim) >= rr.PASS), "drift": rec["drift_cosine"], "norm_ratio": rec.get("norm_ratio")}
    if gallery is not None:
        slots = np.load(rec["recon_path"])["recon_slots"].astype(np.float32)
        queries = torch.as_tensor(slots).clamp(0, 1)
        out["h"], out["picks"] = rr.retrieval_hit(queries, rec["slot_labels"], rec["image_indices"], gallery)
    return out


def build_cells(records, gallery):
    """{(setting, N, E, split, kind): {release: scores}}."""
    cells = {}
    for rec in records:
        key = (rec["setting"], rec["n"], rec["e"], rec["split_name"], rec["attacker"])
        by_release = cells.setdefault(key, {})
        assert rec["release"] not in by_release, f"two records for release {rec['release']} in {key}"
        by_release[rec["release"]] = release_scores(rec, gallery)
    return cells


def _ci(a, b, strata, level, method="t"):
    kw = BOOT if method == "bootstrap" else {}
    return stats.paired_diff_interval(a, b, strata, level=level, method=method, **kw)


def versus_baseline(cell, baselines, n, score="s", base="b"):
    """Paired (score - baseline) per release: 95% t and bootstrap intervals."""
    rels = sorted(cell)
    a = [cell[r][score] for r in rels]
    b = [float(baselines[(n, r)][base]) for r in rels]
    strata = [cell[r]["checkpoint"] for r in rels]
    return {"t95": _ci(a, b, strata, 0.95), "boot95": _ci(a, b, strata, 0.95, "bootstrap"),
            "mean_score": float(np.mean(a)), "mean_baseline": float(np.mean(b)),
            "enough": stats.enough_releases(strata, checkpoints=CHECKPOINTS), "n": len(rels),
            "per_checkpoint": per_checkpoint(cell, score)}


def gap(multi, single, score="s"):
    """Paired multi - single per release (both splits present): 90% and 95% intervals."""
    rels = sorted(set(multi) & set(single))
    a = [multi[r][score] for r in rels]
    b = [single[r][score] for r in rels]
    strata = [multi[r]["checkpoint"] for r in rels]
    out = {"t90": _ci(a, b, strata, 0.90), "t95": _ci(a, b, strata, 0.95),
           "boot90": _ci(a, b, strata, 0.90, "bootstrap"), "n": len(rels),
           "enough": stats.enough_releases(strata, checkpoints=CHECKPOINTS)}
    if score in ("a", "h"):
        pa, pb = np.array(a, bool), np.array(b, bool)
        out["counts"] = {"both": int((pa & pb).sum()), "only multi": int((pa & ~pb).sum()),
                         "only single": int((~pa & pb).sum()), "neither": int((~pa & ~pb).sum())}
    return out


def per_checkpoint(cell, score):
    return {k: float(np.mean([v[score] for v in cell.values() if v["checkpoint"] == k]))
            for k in CHECKPOINTS if any(v["checkpoint"] == k for v in cell.values())}


def label_block(cells, baselines, setting, n, kind, main_kind):
    """Section 12 outcome for (setting, N, attacker kind) at E = 5, with PC(setting, N)."""
    pc_key = (setting, n, 1, f"1x{n}", main_kind)
    single_key = (setting, n, 5, f"1x{n}", kind)
    multi_key = (setting, n, 5, f"{n}x1", kind)
    missing = [k for k in (pc_key, single_key, multi_key) if k not in cells]
    if missing:
        return {"outcome": "not run", "missing": [list(k) for k in missing]}
    pc = versus_baseline(cells[pc_key], baselines, n)
    single = versus_baseline(cells[single_key], baselines, n)
    multi = versus_baseline(cells[multi_key], baselines, n)
    g = gap(cells[multi_key], cells[single_key])
    enough = pc["enough"] and single["enough"] and multi["enough"] and g["enough"]
    outcome = stats.classify_outcome(pc["t95"]["lo"], single["t95"]["lo"], multi["t95"]["lo"],
                                     g["t90"]["lo"], g["t90"]["hi"], enough=enough)
    return {"outcome": outcome, "pc": pc, "single": single, "multi": multi, "gap": g,
            "pc_passes": stats.beats_baseline(pc["t95"]["lo"]),
            "single_bb": stats.beats_baseline(single["t95"]["lo"]),
            "multi_bb": stats.beats_baseline(multi["t95"]["lo"])}


def analyze(records, baselines, gallery, main_objective="cosine"):
    """All numbers of the report, as a dict."""
    assert main_objective in ("cosine", "basic"), main_objective
    main_kind = "cosine" if main_objective == "cosine" else "lbfgs"
    second_kind = "lbfgs" if main_kind == "cosine" else "cosine"
    cells = build_cells(records, gallery)
    for key, cell in cells.items():
        for r in cell:
            assert (key[1], r) in baselines, f"no baseline for N = {key[1]}, release {r}"
    blocks = sorted({(k[0], k[1]) for k in cells}, key=lambda b: (b != MAIN, b[1], b[0]))
    res = {"main_objective": main_objective, "main_kind": main_kind, "second_kind": second_kind,
           "blocks": [list(b) for b in blocks], "labels": {}, "cells": {}, "drift": {}, "gaps": {}}
    has_h = gallery is not None
    for setting, n in blocks:
        for kind in ("unrolled", main_kind, second_kind):
            res["labels"][f"{setting}|{n}|{kind}"] = label_block(cells, baselines, setting, n, kind, main_kind)
        # Drift: one value per release (all E = 5 cells of a release share it).
        e5 = {r: v for k, c in cells.items() if k[:3] == (setting, n, 5) for r, v in c.items()}
        e1 = [v["drift"] for k, c in cells.items() if k[:3] == (setting, n, 1) for v in c.values()]
        if e5:
            d = np.array([v["drift"] for v in e5.values()])
            nr = np.array([v["norm_ratio"] for v in e5.values() if v["norm_ratio"] is not None])
            res["drift"][f"{setting}|{n}"] = {
                "median": float(np.median(d)), "min": float(d.min()), "max": float(d.max()), "n": len(d),
                "per_checkpoint": {k: float(np.median([v["drift"] for v in e5.values() if v["checkpoint"] == k]))
                                   for k in CHECKPOINTS if any(v["checkpoint"] == k for v in e5.values())},
                "norm_ratio_median": float(np.median(nr)) if len(nr) else None,
                "e1_min": float(min(e1)) if e1 else None}
    for key in sorted(cells):
        setting, n, e, split, kind = key
        cell = cells[key]
        entry = {"n_releases": len(cell), "mean_ssim": float(np.mean([v["s"] for v in cell.values()])),
                 "any_pass_rate": float(np.mean([v["a"] for v in cell.values()])),
                 "ssim_vs_baseline": versus_baseline(cell, baselines, n, "s", "b"),
                 "any_pass_vs_baseline": versus_baseline(cell, baselines, n, "a", "baseline_any_pass"),
                 "per_checkpoint_ssim": per_checkpoint(cell, "s")}
        if has_h:
            entry["hit_rate"] = float(np.mean([v["h"] for v in cell.values()]))
            entry["hit_vs_baseline"] = versus_baseline(cell, baselines, n, "h", "baseline_hit")
            entry["per_checkpoint_hit"] = per_checkpoint(cell, "h")
        res["cells"]["|".join(map(str, key))] = entry
        if e == 5 and split == f"{n}x1" and (setting, n, 5, f"1x{n}", kind) in cells:
            single = cells[(setting, n, 5, f"1x{n}", kind)]
            res["gaps"][f"{setting}|{n}|{kind}"] = {
                "ssim": gap(cell, single, "s"), "any_pass": gap(cell, single, "a"),
                **({"hit": gap(cell, single, "h")} if has_h else {})}
    # Baselines side by side, per N (shared by all cells of that N).
    res["baselines"] = {}
    for n in sorted({k[1] for k in cells}):
        rels = sorted({r for k, c in cells.items() if k[1] == n for r in c})
        bs = [baselines[(n, r)] for r in rels]
        res["baselines"][str(n)] = {
            "releases": len(bs), "mean_b": float(np.mean([b["b"] for b in bs])),
            "mean_A": float(np.mean([b["A"] for b in bs])), "mean_S": float(np.mean([b["S"] for b in bs])),
            "any_pass_rate": float(np.mean([b["baseline_any_pass"] for b in bs])),
            "hit_rate": float(np.mean([b["baseline_hit"] for b in bs])),
            "hit_exact_chance": float(np.mean([b["hit_exact_chance"] for b in bs])),
            "context": {k: float(np.mean([b["context_mean_ssim"][k] for b in bs]))
                        for k in bs[0].get("context_mean_ssim", {})}}
    main_key = f"{MAIN[0]}|{MAIN[1]}|unrolled"
    res["main"] = res["labels"].get(main_key, {"outcome": "not run"})
    return res


def _f(x, d=3):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def _iv(ci, d=3):
    return f"{_f(ci['mean'], d)} [{_f(ci['lo'], d)}, {_f(ci['hi'], d)}]"


def _pc(d):
    return ", ".join(f"k{k}: {v:.3f}" for k, v in sorted(d.items(), key=lambda kv: int(kv[0])))


def _cell_name(key):
    setting, n, e, split, kind = key.split("|")
    return f"{setting} | {n} | {e} | {split} | {KIND_LABEL[kind]}"


SENTENCE = {stats.LESS: "gave worse reconstructions than", stats.MORE: "gave better reconstructions than",
            stats.SAME: "gave reconstructions within ±0.05 SSIM of"}


def render(res, title="Journal results"):
    L = [f"# {title}", "",
         "*Made by `scripts/analyze_journal.py`. "
         f"Main objective of attacker 1: **{res['main_objective']}**. Attacker 2 uses cosine.*", ""]
    m = res["main"]
    L += ["## 1. Main endpoint", "", "MNIST fresh, N = 2, E = 5, attacker 2 (cosine). "
          "Positive control: attacker 1, main objective, E = 1.", ""]
    if m["outcome"] == "not run":
        L += ["**Outcome: not run** (cells missing).", ""]
    else:
        g = m["gap"]["t90"]
        L += [f"**Outcome: {m['outcome']}**", "",
              "| Part | Estimate [interval] | Rule | Result |", "|---|---|---|---|",
              f"| Positive control (s − b, E = 1) | {_iv(m['pc']['t95'])} (95%) | lower end > 0 | "
              f"{'passes' if m['pc_passes'] else 'fails'} |",
              f"| 1 × 2 beats baseline (s − b) | {_iv(m['single']['t95'])} (95%) | lower end > 0 | "
              f"{'yes' if m['single_bb'] else 'no'} |",
              f"| 2 × 1 beats baseline (s − b) | {_iv(m['multi']['t95'])} (95%) | lower end > 0 | "
              f"{'yes' if m['multi_bb'] else 'no'} |",
              f"| Gap 2 × 1 − 1 × 2 | {_iv(g)} (90%) | margin ±0.05 | n = {m['gap']['n']} |", "",
              f"Bootstrap check (report only): gap 90% {_iv(m['gap']['boot90'])}; "
              f"control 95% {_iv(m['pc']['boot95'])}.", ""]
        if m["outcome"] in SENTENCE:
            L += [f"Sentence: \"Under attacker 2, on MNIST fresh at N = 2 and E = 5, the 2-client sum "
                  f"{SENTENCE[m['outcome']]} the 1-client update (gap {_f(g['mean'])}, 90% interval "
                  f"[{_f(g['lo'])}, {_f(g['hi'])}], {m['gap']['n']} paired releases on 3 checkpoints).\"", ""]
    L += ["## 2. Secondary labels (E = 5)", "",
          "Same rules, each (setting, N) with its own positive control (attacker 1, main objective, E = 1). "
          "Not adjusted for multiple tests.", "",
          "| Setting | N | Attacker | Outcome | PC s − b (95%) | 1×N s − b (95%) | N×1 s − b (95%) | Gap (90%) |",
          "|---|---|---|---|---|---|---|---|"]
    for key, lab in res["labels"].items():
        setting, n, kind = key.split("|")
        if lab["outcome"] == "not run":
            L.append(f"| {setting} | {n} | {KIND_LABEL[kind]} | not run | – | – | – | – |")
            continue
        L.append(f"| {setting} | {n} | {KIND_LABEL[kind]} | {lab['outcome']} | {_iv(lab['pc']['t95'])} | "
                 f"{_iv(lab['single']['t95'])} | {_iv(lab['multi']['t95'])} | {_iv(lab['gap']['t90'])} |")
    L += ["", "## 3. Second endpoint: any image passes (SSIM ≥ 0.9)", "",
          "| Setting | N | E | Split | Attacker | Rate | Rate − baseline (95%) |", "|---|---|---|---|---|---|---|"]
    for key, c in res["cells"].items():
        L.append(f"| {_cell_name(key)} | {c['any_pass_rate']:.3f} | {_iv(c['any_pass_vs_baseline']['t95'])} |")
    L += ["", "| Setting | N | Attacker | Gap N×1 − 1×N (90%) | (95%) | Both / only N×1 / only 1×N / neither |",
          "|---|---|---|---|---|---|"]
    for key, g in res["gaps"].items():
        setting, n, kind = key.split("|")
        c = g["any_pass"]["counts"]
        L.append(f"| {setting} | {n} | {KIND_LABEL[kind]} | {_iv(g['any_pass']['t90'])} | {_iv(g['any_pass']['t95'])} "
                 f"| {c['both']} / {c['only multi']} / {c['only single']} / {c['neither']} |")
    hit = any("hit_rate" in c for c in res["cells"].values())
    L += ["", "## 4. Averages hide the risk", "",
          "| Setting | N | E | Split | Attacker | Releases | Mean SSIM | Any passes | Retrieval hit | "
          "SSIM per checkpoint |", "|---|---|---|---|---|---|---|---|---|---|"]
    for key, c in res["cells"].items():
        L.append(f"| {_cell_name(key)} | {c['n_releases']} | {c['mean_ssim']:.3f} | {c['any_pass_rate']:.3f} | "
                 f"{c['hit_rate']:.3f} | {_pc(c['per_checkpoint_ssim'])} |" if hit else
                 f"| {_cell_name(key)} | {c['n_releases']} | {c['mean_ssim']:.3f} | {c['any_pass_rate']:.3f} | – | "
                 f"{_pc(c['per_checkpoint_ssim'])} |")
    for n, b in res["baselines"].items():
        L.append(f"| label-aware baseline | {n} | – | – | max(class average, same class) | {b['releases']} | "
                 f"{b['mean_b']:.3f} | {b['any_pass_rate']:.3f} | {b['hit_rate']:.3f} | – |")
    L += ["", "Baseline parts and context (mean SSIM): " + "; ".join(
        f"N = {n}: class average {b['mean_A']:.3f}, same class {b['mean_S']:.3f}, exact retrieval chance "
        f"{b['hit_exact_chance']:.4f}, " + ", ".join(f"{k} {v:.3f}" for k, v in b["context"].items())
        for n, b in res["baselines"].items()), ""]
    L += ["## 5. Drift (E = 5)", "", "| Setting | N | Releases | Median | Range | Median per checkpoint | "
          "Median norm ratio | E = 1 drift, min |", "|---|---|---|---|---|---|---|---|"]
    for key, d in res["drift"].items():
        setting, n = key.split("|")
        L.append(f"| {setting} | {n} | {d['n']} | {d['median']:.4f} | {d['min']:.4f}–{d['max']:.4f} | "
                 f"{_pc(d['per_checkpoint'])} | {_f(d['norm_ratio_median'], 4)} | {_f(d['e1_min'], 6)} |")
    sk = res["second_kind"]
    L += ["", f"## 6. Secondary objective ({KIND_LABEL[sk]})", "",
          "| Setting | N | E | Split | Mean SSIM | s − b (95%) | Any passes |", "|---|---|---|---|---|---|---|"]
    for key, c in res["cells"].items():
        if key.endswith(f"|{sk}"):
            setting, n, e, split, _ = key.split("|")
            L.append(f"| {setting} | {n} | {e} | {split} | {c['mean_ssim']:.3f} | "
                     f"{_iv(c['ssim_vs_baseline']['t95'])} | {c['any_pass_rate']:.3f} |")
    if hit:
        L += ["", "## 7. Retrieval (top-1 in the slot label's class gallery)", "",
              "| Setting | N | E | Split | Attacker | Hit rate | Hit − baseline (95%) | Per checkpoint |",
              "|---|---|---|---|---|---|---|---|"]
        for key, c in res["cells"].items():
            L.append(f"| {_cell_name(key)} | {c['hit_rate']:.3f} | {_iv(c['hit_vs_baseline']['t95'])} | "
                     f"{_pc(c['per_checkpoint_hit'])} |")
        L += ["", "| Setting | N | Attacker | Hit gap N×1 − 1×N (90%) | (95%) | Both / only N×1 / only 1×N / neither |",
              "|---|---|---|---|---|---|"]
        for key, g in res["gaps"].items():
            setting, n, kind = key.split("|")
            c = g["hit"]["counts"]
            L.append(f"| {setting} | {n} | {KIND_LABEL[kind]} | {_iv(g['hit']['t90'])} | {_iv(g['hit']['t95'])} | "
                     f"{c['both']} / {c['only multi']} / {c['only single']} / {c['neither']} |")
    L += ["", "Settings (fresh vs trained) use the same releases; any comparison between them is descriptive."]
    return "\n".join(L) + "\n"


def mnist_gallery():
    from src.data import load_datasets
    _, test_ds, _ = load_datasets("mnist", normalize=False)
    images = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    labels = np.array([int(y) for y in test_ds.targets])
    return rr.Gallery(images, labels, np.arange(len(labels)))


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--records", default="results/journal/targets")
    p.add_argument("--baselines", default="results/journal/baselines_mnist.json")
    p.add_argument("--objective", default="results/journal/pilot/objective.json",
                   help="the pilot's objective rule (main objective of attacker 1)")
    p.add_argument("--main-objective", choices=["cosine", "basic"], default=None, help="override --objective")
    p.add_argument("--no-retrieval", action="store_true", help="skip retrieval (no MNIST gallery needed)")
    p.add_argument("--title", default="Journal results")
    p.add_argument("--out", default="results/journal/report")
    args = p.parse_args()
    torch.set_num_threads(4)
    main_objective = args.main_objective or main_objective_from(args.objective)
    if main_objective is None:
        sys.exit(f"{args.objective} not found: run the pilot first, or pass --main-objective")
    records = load_records(args.records)
    if not records:
        sys.exit(f"no records under {args.records}")
    baselines, _ = load_baselines(args.baselines)
    gallery = None if args.no_retrieval else mnist_gallery()
    res = analyze(records, baselines, gallery, main_objective)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(_jsonable(res), f, indent=1)
    with open(args.out + ".md", "w") as f:
        f.write(render(res, args.title))
    print(f"main endpoint: {res['main']['outcome']}")
    print(f"Saved -> {args.out}.md, {args.out}.json")


if __name__ == "__main__":
    main()
