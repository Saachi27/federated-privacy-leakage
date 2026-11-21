"""Timing projection for the journal target run, from the pilot."""

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.attack_multiclient import ATTACKERS, CUTS, cells

RELEASES = 30
NIGHTS = 5
BLOCKS = [("mnist-fresh", 2), ("mnist-trained", 2), ("mnist-fresh", 4), ("mnist-trained", 4)]


def cell_key(e, split, attacker):
    return f"E{e}_{split[0]}x{split[1]}_{attacker}"


def load(grid):
    """{(setting, N): {cell: [seconds]}} from pilot records."""
    out = {}
    for path in glob.glob(os.path.join(grid, "*", "N*", "E*_r*.json")):
        with open(path) as f:
            r = json.load(f)
        out.setdefault((r["setting"], r["n"]), {}).setdefault(
            cell_key(r["e"], r["split"], r["attacker"]), []).append(r["seconds"])
    return out


def per_release(times, setting, n):
    """Mean worker-seconds for one release of (setting, N), all cells, and where it came from."""
    if (setting, n) in times:
        src, t = f"measured ({setting}, N = {n})", times[(setting, n)]
    elif ("mnist-fresh", n) in times:
        src, t = f"= MNIST fresh, N = {n}", times[("mnist-fresh", n)]
    else:
        return None, "not timed"
    want = [cell_key(e, s, a) for e, s, a in cells(n, [1, 5], ["single", "multi"], ATTACKERS)]
    missing = [c for c in want if c not in t]
    if missing:
        return None, f"cells not timed: {', '.join(missing)}"
    return float(sum(np.mean(t[c]) for c in want)), src


def project(times, workers, packing, night_hours):
    rows = []
    for setting, n in BLOCKS:
        sec, src = per_release(times, setting, n)
        rows.append({"setting": setting, "n": n, "per_release": sec, "source": src,
                     "worker_hours": None if sec is None else sec * RELEASES / 3600})
    levels = []
    for cut in range(len(CUTS) + 1):
        dropped = set(CUTS[:cut])
        kept = [r for r in rows if (r["setting"], r["n"]) not in dropped]
        if any(r["worker_hours"] is None for r in kept):
            levels.append({"cut": cut, "worker_hours": None})
            continue
        wh = sum(r["worker_hours"] for r in kept)
        wall = wh / workers / packing
        levels.append({"cut": cut, "worker_hours": wh, "wall_hours": wall, "nights": wall / night_hours,
                       "fits": wall <= NIGHTS * night_hours, "blocks": [(r["setting"], r["n"]) for r in kept]})
    return rows, levels


def render(times, rows, levels, workers, packing, night_hours, wall, n2_tasks):
    L = ["# Pilot timing and projection", "",
         f"*Made by `scripts/pilot_timing.py` from the pilot records (tuning images only). "
         f"{workers} workers, 1 torch thread each, 2 restarts, fixed steps.*", "",
         "## Measured: mean worker-seconds per task", "",
         "| Setting | N | Cell | Tasks timed | Mean s | Min s | Max s |", "|---|---|---|---|---|---|---|"]
    for (setting, n), t in sorted(times.items()):
        for c in sorted(t):
            v = t[c]
            L.append(f"| {setting} | {n} | {c} | {len(v)} | {np.mean(v):.0f} | {min(v):.0f} | {max(v):.0f} |")
    L += ["", f"Pilot N = 2 wall time: {wall / 60:.1f} min for {n2_tasks} tasks "
              f"(packing {packing:.2f}: share of {workers} workers kept busy)." if wall else "", "",
          "## One release, all cells (worker-seconds)", "",
          "| Setting | N | Per release | 30 releases (worker-hours) | Source |", "|---|---|---|---|---|"]
    for r in rows:
        per = "–" if r["per_release"] is None else "%.0f s" % r["per_release"]
        wh = "–" if r["worker_hours"] is None else "%.1f" % r["worker_hours"]
        L.append(f"| {r['setting']} | {r['n']} | {per} | {wh} | {r['source']} |")
    L += ["", f"## Full target grid at each cut level ({night_hours:g}-hour nights, 5 nights = "
              f"{NIGHTS * night_hours:g} h)", "",
          "| Cut level | Blocks kept | Worker-hours | Wall hours | Nights | Fits in 5 nights? |",
          "|---|---|---|---|---|---|"]
    names = {0: "none", 1: "1 (drop trained N=4)", 2: "2 (also fresh N=4)", 3: "3 (also trained N=2)"}
    for lv in levels:
        if lv["worker_hours"] is None:
            L.append(f"| {names[lv['cut']]} | – | – | – | – | not timed |")
            continue
        kept = ", ".join(f"{s.replace('mnist-', '')} N={n}" for s, n in lv["blocks"])
        L.append(f"| {names[lv['cut']]} | {kept} | {lv['worker_hours']:.1f} | {lv['wall_hours']:.1f} | "
                 f"{lv['nights']:.2f} | {'yes' if lv['fits'] else 'no'} |")
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--grid", default="results/journal/pilot/grid")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--wall", type=float, default=None,
                   help="wall seconds of the timed N = 2 pilot (all segments); sets the packing")
    p.add_argument("--night-hours", type=float, default=10.0)
    p.add_argument("--notes", default=None, help="a markdown file appended to the output")
    p.add_argument("--out", default="results/journal/pilot/timing.md")
    args = p.parse_args()
    times = load(args.grid)
    n2 = times.get(("mnist-fresh", 2), {})
    n2_tasks = sum(len(v) for v in n2.values())
    packing = 1.0
    if args.wall:
        packing = min(1.0, sum(sum(v) for v in n2.values()) / (args.workers * args.wall))
    rows, levels = project(times, args.workers, packing, args.night_hours)
    md = render(times, rows, levels, args.workers, packing, args.night_hours, args.wall, n2_tasks)
    if args.notes:
        with open(args.notes) as f:
            md += "\n" + f.read()
    with open(args.out, "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
