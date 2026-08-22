"""Aggregate the encoder x k grid into one table.

    python -m eventtok.scripts.report_encoder_sweep

Reports the *whole* grid, not the best cell. The point of the sweep is to find out
whether the encoder ordering is a property of the encoders or of one cluster count,
and that question is only answered by cells where the ordering breaks. A best-cell
summary would hide exactly the evidence the sweep was run to collect.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from .. import paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default=None)
    args = ap.parse_args()

    pattern = args.pattern or str(paths.CACHE_ROOT / "eval" / "enck_*_k*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no results matching {pattern}")

    by_task: dict[str, dict[int, dict]] = {}
    for path in files:
        with open(path) as fh:
            payload = json.load(fh)
        rows = {r["condition"]: r for r in payload["results"]}
        by_task.setdefault(payload["task"], {})[payload["k"]] = {
            "rows": rows,
            "majority": next(iter(rows.values()))["majority"],
        }

    for task, cells in by_task.items():
        ks = sorted(cells)
        conditions = list(next(iter(cells.values()))["rows"])
        width = max(len(c) for c in conditions)
        print(f"\n=== {task} ===  majority {cells[ks[0]]['majority']:.1%}")
        print(f"{'condition':<{width}} " + "".join(f"{'k=' + str(k):>9}" for k in ks))
        for cond in conditions:
            cells_str = "".join(
                f"{cells[k]['rows'][cond]['accuracy']:>8.1%} " if cond in cells[k]["rows"]
                else f"{'-':>9}"
                for k in ks
            )
            print(f"{cond:<{width}} {cells_str}")

        # Whether the headline ordering survives every k is the actual result.
        print("  per-k best condition:")
        flips = set()
        for k in ks:
            rows = cells[k]["rows"]
            best = max(rows.values(), key=lambda r: r["accuracy"])
            flips.add(best["condition"])
            print(f"    k={k:<4} {best['condition']:<30} {best['accuracy']:.1%}")
        if len(flips) > 1:
            print(
                f"  the best condition CHANGES with k ({len(flips)} different winners) "
                f"-- do not report a single row as the result"
            )
        else:
            print(f"  same winner at every k: {next(iter(flips))}")

        # DINOv2 vs SigLIP, matched on everything except the encoder.
        print("  dinov2l - siglip, matched condition:")
        for a, b in [("dinov2l only", "siglip only"),
                     ("action + dinov2l", "action + siglip")]:
            deltas = [
                cells[k]["rows"][a]["accuracy"] - cells[k]["rows"][b]["accuracy"]
                for k in ks
                if a in cells[k]["rows"] and b in cells[k]["rows"]
            ]
            if deltas:
                sign = "all positive" if min(deltas) > 0 else (
                    "all negative" if max(deltas) < 0 else "MIXED SIGN")
                print(
                    f"    {a:<20} vs {b:<20} "
                    + " ".join(f"{d:+6.1%}" for d in deltas)
                    + f"   [{sign}]"
                )


if __name__ == "__main__":
    main()
