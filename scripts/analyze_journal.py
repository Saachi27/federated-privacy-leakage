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
    out = {}
    for n, recs in data["releases"].items():
        seen = set()
        for rec in recs:
            key = (int(n), rec["release"])
            if key in out:
                raise PairingError(f"baselines: two records for N = {n}, release {rec['release']}")
            if "image_indices" not in rec:
                raise PairingError(f"baselines: N = {n}, release {rec['release']} has no image_indices")
            idx = tuple(rec["image_indices"])
            if idx in seen:
                raise PairingError(f"baselines: N = {n}, image indices {list(idx)} appear twice")
            seen.add(idx)
            out[key] = rec
    return out, data


def main_objective_from(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)["main_objective"]
    return None


class PairingError(ValueError):
    """Two records that should be the same release are not: stop."""


PAIR_FIELDS = ("release", "image_indices", "checkpoint", "checkpoint_hash")


def check_pair(x, y, what, fields=PAIR_FIELDS):
    """Paired records must share release id, image indices, checkpoint and its hash."""
    for f in fields:
        if f not in x or f not in y:
            raise PairingError(f"{what}: release {x.get('release')} has no {f} on one side")
        if x[f] != y[f]:
            raise PairingError(f"{what}: release {x.get('release')}: {f} differs ({x[f]} vs {y[f]})")


def baseline_of(baselines, n, scores, what="baseline"):
    """The baseline record of this release, joined by (N, release) and image indices."""
    key = (n, scores["release"])
    if key not in baselines:
        raise PairingError(f"{what}: no baseline for N = {n}, release {scores['release']}")
    b = baselines[key]
    check_pair(scores, b, f"{what} N = {n}", fields=("release", "image_indices"))
    return b


def release_scores(rec, gallery):
    """S_r, a_r and h_r for one record."""
    ssim = [float(v) for v in rec["ssim"]]
    out = {"release": rec["release"], "checkpoint": rec["checkpoint"],
           "checkpoint_hash": rec.get("checkpoint_hash"), "image_indices": [int(i) for i in rec["image_indices"]],
           "labels": [int(c) for c in rec["labels"]], "s": float(np.mean(ssim)),
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
        if "checkpoint_hash" not in rec:
            raise PairingError(f"{key} release {rec['release']}: record has no checkpoint_hash")
        by_release[rec["release"]] = release_scores(rec, gallery)
    check_cells_paired(cells)
    return cells


def check_cells_paired(cells):
    """Every cell of a (setting, N) must hold the same release on the same images and checkpoint."""
    first = {}
    for key, cell in sorted(cells.items()):
        for r, v in cell.items():
            ref = first.setdefault((key[0], key[1], r), (key, v))
            check_pair(v, ref[1], f"{key} vs {ref[0]}")


def _ci(a, b, strata, level, method="t"):
    kw = BOOT if method == "bootstrap" else {}
    return stats.paired_diff_interval(a, b, strata, level=level, method=method, **kw)


def paired_part(a, b, strata, levels):
    """Counts per checkpoint first, then the intervals."""
    counts = {k: 0 for k in CHECKPOINTS}
    counts.update(stats.releases_per_checkpoint(strata) if len(strata) else {})
    out = {"n": len(a), "per_checkpoint_n": counts,
           "enough": stats.enough_releases(strata, checkpoints=CHECKPOINTS)}
    if not len(a):
        out["status"] = stats.NO_PAIRS
        out["mean_diff"] = None
        for lv in levels:
            out[f"t{lv}"] = out[f"boot{lv}"] = None
        return out
    out["status"] = "decides" if out["enough"] else stats.DESCRIPTIVE
    for lv in levels:
        out[f"t{lv}"] = _ci(a, b, strata, lv / 100) if out["enough"] else None
        out[f"boot{lv}"] = _ci(a, b, strata, lv / 100, "bootstrap")
    out["mean_diff"] = out[f"boot{levels[0]}"]["mean"]
    return out


def versus_baseline(cell, baselines, n, score="s", base="b"):
    """Paired (score - baseline) per release: 95% t (if enough releases) and bootstrap."""
    rels = sorted(cell)
    a = [cell[r][score] for r in rels]
    b = [float(baseline_of(baselines, n, cell[r])[base]) for r in rels]
    strata = [cell[r]["checkpoint"] for r in rels]
    return {**paired_part(a, b, strata, (95,)),
            "mean_score": float(np.mean(a)) if a else None, "mean_baseline": float(np.mean(b)) if b else None,
            "per_checkpoint": per_checkpoint(cell, score)}


def passes(part):
    """Beats-baseline / positive-control result: True/False, or None if descriptive only."""
    return stats.beats_baseline(part["t95"]["lo"]) if part["status"] == "decides" else None


def gap(multi, single, score="s"):
    """Paired multi - single per release (both splits present): 90% and 95% intervals."""
    rels = sorted(set(multi) & set(single))
    for r in rels:
        check_pair(multi[r], single[r], "gap")
    a = [multi[r][score] for r in rels]
    b = [single[r][score] for r in rels]
    strata = [multi[r]["checkpoint"] for r in rels]
    out = paired_part(a, b, strata, (90, 95))
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
    # Counts are checked before any interval is read.
    enough = all(p["status"] == "decides" for p in (pc, single, multi, g))
    if enough:
        outcome = stats.classify_outcome(pc["t95"]["lo"], single["t95"]["lo"], multi["t95"]["lo"],
                                         g["t90"]["lo"], g["t90"]["hi"])
    else:
        outcome = stats.DESCRIPTIVE
    return {"outcome": outcome, "pc": pc, "single": single, "multi": multi, "gap": g,
            "pc_passes": passes(pc) if enough else None,
            "single_bb": passes(single) if enough else None,
            "multi_bb": passes(multi) if enough else None}


SAME_LABEL, DIFF_LABEL = "same label", "different labels"


def same_label(scores):
    return len(set(scores["labels"])) == 1


def label_split(multi, single):
    """The gap split by same-label vs different-label releases."""
    out = {}
    for name, want in ((SAME_LABEL, True), (DIFF_LABEL, False)):
        m = {r: v for r, v in multi.items() if same_label(v) == want}
        s = {r: v for r, v in single.items() if same_label(v) == want}
        g = gap(m, s)
        paired = sorted(set(m) & set(s))
        g["releases"] = paired
        g["median_drift"] = float(np.median([m[r]["drift"] for r in paired])) if paired else None
        out[name] = g
    return out


def drift_per_release(cells, setting, n):
    """One row per release at E = 5: checkpoint, labels, drift, norm ratio."""
    rows = {}
    for k, c in cells.items():
        if k[:3] == (setting, n, 5):
            for r, v in c.items():
                rows.setdefault(r, {"release": r, "checkpoint": v["checkpoint"], "labels": v["labels"],
                                    "same_label": same_label(v), "drift": v["drift"],
                                    "norm_ratio": v["norm_ratio"]})
    return [rows[r] for r in sorted(rows)]


def expected_cells(setting, n):
    """The grid of one (setting, N) block, as scripts/attack_multiclient.py runs it."""
    return [(setting, n, 1, f"1x{n}", k) for k in ("lbfgs", "cosine")] + \
        [(setting, n, 5, sp, k) for sp in (f"1x{n}", f"{n}x1") for k in ("lbfgs", "cosine", "unrolled")]


def missing_releases(cells, baselines, blocks):
    """{cell key: release ids in the baseline file for that N but not in the cell}."""
    out = {}
    for setting, n in blocks:
        want = sorted(r for (bn, r) in baselines if bn == n)
        for key in expected_cells(setting, n):
            have = cells.get(key, {})
            out["|".join(map(str, key))] = {"expected": len(want), "have": len(have),
                                             "missing": [r for r in want if r not in have]}
    return out


def analyze(records, baselines, gallery, main_objective="cosine", gallery_sizes=None):
    """All numbers of the report, as a dict."""
    assert main_objective in ("cosine", "basic"), main_objective
    main_kind = "cosine" if main_objective == "cosine" else "lbfgs"
    second_kind = "lbfgs" if main_kind == "cosine" else "cosine"
    cells = build_cells(records, gallery)
    for key, cell in cells.items():
        for v in cell.values():
            baseline_of(baselines, key[1], v, f"baseline for {key}")
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
        bs = [baselines[(n, r)] for r in rels]  # each already joined and checked above
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
    # Secondary; computed after the main outcome and never fed back.
    res["drift_releases"] = {f"{s}|{n}": drift_per_release(cells, s, n) for s, n in blocks}
    mk, sk = (*MAIN, 5, f"{MAIN[1]}x1", "unrolled"), (*MAIN, 5, f"1x{MAIN[1]}", "unrolled")
    res["label_split"] = label_split(cells[mk], cells[sk]) if mk in cells and sk in cells else None
    res["missing_releases"] = missing_releases(cells, baselines, blocks)
    # Every class gallery size is reported.
    file_sizes = {int(c): int(v) for c, v in gallery_sizes.items()} if gallery_sizes else None
    used = {int(c): int(v) for c, v in gallery.sizes().items()} if gallery is not None else None
    if used and file_sizes and used != file_sizes:
        raise ValueError(f"gallery sizes {used} differ from the baseline file's {file_sizes}")
    res["gallery_sizes"] = used or file_sizes
    return res


def _f(x, d=3):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def _iv(ci, d=3):
    return f"{_f(ci['mean'], d)} [{_f(ci['lo'], d)}, {_f(ci['hi'], d)}]"


def _n(p):
    return ", ".join(f"k{k}: {v}" for k, v in sorted(p["per_checkpoint_n"].items(), key=lambda kv: int(kv[0])))


def _part(p, lv=95):
    """A paired part: its t-interval if it decides, else the numbers only."""
    if p["status"] == stats.NO_PAIRS:
        return "no paired releases"
    if p["status"] != "decides":
        return f"descriptive only: {_iv(p[f'boot{lv}'])} bootstrap; releases {_n(p)}"
    return _iv(p[f"t{lv}"])


def _yes(v, yes="yes", no="no"):
    return "descriptive only (no pass/fail)" if v is None else (yes if v else no)


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
              f"| Positive control (s − b, E = 1) | {_part(m['pc'])} (95%) | lower end > 0 | "
              f"{_yes(m['pc_passes'], 'passes', 'fails')} |",
              f"| 1 × 2 beats baseline (s − b) | {_part(m['single'])} (95%) | lower end > 0 | "
              f"{_yes(m['single_bb'])} |",
              f"| 2 × 1 beats baseline (s − b) | {_part(m['multi'])} (95%) | lower end > 0 | "
              f"{_yes(m['multi_bb'])} |",
              f"| Gap 2 × 1 − 1 × 2 | {_part(m['gap'], 90)} (90%) | margin ±0.05 | n = {m['gap']['n']} |", ""]
        if m["gap"]["status"] != stats.NO_PAIRS and m["pc"]["status"] != stats.NO_PAIRS:
            L += [f"Bootstrap check (report only): gap 90% {_iv(m['gap']['boot90'])}; "
                  f"control 95% {_iv(m['pc']['boot95'])}.", ""]
        if m["outcome"] == stats.DESCRIPTIVE:
            L += ["Fewer than 5 paired releases on some checkpoint: numbers only, "
                  "no outcome label and no pass/fail.", ""]
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
        L.append(f"| {setting} | {n} | {KIND_LABEL[kind]} | {lab['outcome']} | {_part(lab['pc'])} | "
                 f"{_part(lab['single'])} | {_part(lab['multi'])} | {_part(lab['gap'], 90)} |")
    L += ["", "## 3. Second endpoint: any image passes (SSIM ≥ 0.9)", "",
          "| Setting | N | E | Split | Attacker | Rate | Rate − baseline (95%) |", "|---|---|---|---|---|---|---|"]
    for key, c in res["cells"].items():
        L.append(f"| {_cell_name(key)} | {c['any_pass_rate']:.3f} | {_part(c['any_pass_vs_baseline'])} |")
    L += ["", "| Setting | N | Attacker | Gap N×1 − 1×N (90%) | (95%) | Both / only N×1 / only 1×N / neither |",
          "|---|---|---|---|---|---|"]
    for key, g in res["gaps"].items():
        setting, n, kind = key.split("|")
        c = g["any_pass"]["counts"]
        L.append(f"| {setting} | {n} | {KIND_LABEL[kind]} | {_part(g['any_pass'], 90)} | {_part(g['any_pass'])} "
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
                     f"{_part(c['ssim_vs_baseline'])} | {c['any_pass_rate']:.3f} |")
    sizes = res.get("gallery_sizes")
    size_line = ("Class gallery sizes (every test image of the class): " + ", ".join(
        f"class {c}: {v:,}" for c, v in sorted(sizes.items(), key=lambda kv: int(kv[0]))) + ".") if sizes else \
        "Class gallery sizes: not known (no gallery and none in the baseline file)."
    if not hit:
        L += ["", "## 7. Retrieval", "", "Retrieval not computed in this report.", "", size_line]
    if hit:
        L += ["", "## 7. Retrieval (top-1 in the slot label's class gallery)", "", size_line, "",
              "| Setting | N | E | Split | Attacker | Hit rate | Hit − baseline (95%) | Per checkpoint |",
              "|---|---|---|---|---|---|---|---|"]
        for key, c in res["cells"].items():
            L.append(f"| {_cell_name(key)} | {c['hit_rate']:.3f} | {_part(c['hit_vs_baseline'])} | "
                     f"{_pc(c['per_checkpoint_hit'])} |")
        L += ["", "| Setting | N | Attacker | Hit gap N×1 − 1×N (90%) | (95%) | Both / only N×1 / only 1×N / neither |",
              "|---|---|---|---|---|---|"]
        for key, g in res["gaps"].items():
            setting, n, kind = key.split("|")
            c = g["hit"]["counts"]
            L.append(f"| {setting} | {n} | {KIND_LABEL[kind]} | {_part(g['hit'], 90)} | {_part(g['hit'])} | "
                     f"{c['both']} / {c['only multi']} / {c['only single']} / {c['neither']} |")
    L += ["", "## 8. Drift and labels (pre-declared secondary)", "",
          "On the fresh model a same-label pair barely drifts, so it acts like E = 1. Secondary: "
          "this never changes the main outcome.", ""]
    ls = res.get("label_split")
    if ls is None:
        L += ["Main gap split by labels: not run (cells missing).", ""]
    else:
        L += ["Main gap (MNIST fresh, N = 2, E = 5, attacker 2), 2 × 1 − 1 × 2, split by the release's labels:", "",
              "| Releases | Paired n | Per checkpoint | Median drift | Gap (90%) | Gap (95%) |",
              "|---|---|---|---|---|---|"]
        for name, g in ls.items():
            L.append(f"| {name} | {g['n']} | {_n(g)} | {_f(g['median_drift'], 4)} | {_part(g, 90)} | "
                     f"{_part(g)} |")
        L.append("")
    for key, rows in res.get("drift_releases", {}).items():
        setting, n = key.split("|")
        if not rows:
            continue
        L += [f"Drift per release, {setting}, N = {n} (E = 5):", "",
              "| Release | Checkpoint | Labels | Same label | Drift | Norm ratio |", "|---|---|---|---|---|---|"]
        L += [f"| {r['release']} | {r['checkpoint']} | {', '.join(map(str, r['labels']))} | "
              f"{'yes' if r['same_label'] else 'no'} | {r['drift']:.4f} | {_f(r['norm_ratio'], 4)} |" for r in rows]
        L.append("")
    L += ["## 9. Releases per cell", "",
          "Expected: every release in the baseline file for that N. No release is dropped after a result is seen.", "",
          "| Setting | N | E | Split | Attacker | Releases | Of | Missing release ids |", "|---|---|---|---|---|---|---|---|"]
    for key, m in res.get("missing_releases", {}).items():
        miss = "none" if not m["missing"] else ("all (no records)" if not m["have"] else
                                                 ", ".join(map(str, m["missing"])))
        L.append(f"| {_cell_name(key)} | {m['have']} | {m['expected']} | {miss} |")
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
    baselines, base_data = load_baselines(args.baselines)
    gallery = None if args.no_retrieval else mnist_gallery()
    res = analyze(records, baselines, gallery, main_objective, base_data.get("gallery_sizes"))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(_jsonable(res), f, indent=1)
    with open(args.out + ".md", "w") as f:
        f.write(render(res, args.title))
    print(f"main endpoint: {res['main']['outcome']}")
    print(f"Saved -> {args.out}.md, {args.out}.json")


if __name__ == "__main__":
    main()
