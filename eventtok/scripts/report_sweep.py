"""Aggregate a sweep JSONL into a table.

    python -m eventtok.scripts.report_sweep --task SwingXtimes

Prints the whole grid, not a best cell. K and min_frequency interact, so a single
strong combination is easy to over-read; seeing the surface tells you whether an
optimum is real or a one-cell fluke.
"""

from __future__ import annotations

import argparse
import json

from .. import paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="SwingXtimes")
    ap.add_argument("--path", default=None)
    args = ap.parse_args()

    path = args.path or paths.CACHE_ROOT / "eval" / f"sweep_kmeans_{args.task}.jsonl"
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("no rows")
        return

    # Later rows win, so a rerun of one cell supersedes the earlier attempt.
    latest = {(r["clusters"], r["min_frequency"]): r for r in rows}
    ks = sorted({k for k, _ in latest})
    mfs = sorted({m for _, m in latest})

    print(f"{args.task}: {len(latest)} cells of {len(ks)}x{len(mfs)}\n")
    print("count accuracy (exact N / episodes)")
    print("     mf " + " ".join(f"{m:>7d}" for m in mfs))
    for k in ks:
        cells = []
        for m in mfs:
            r = latest.get((k, m))
            cells.append(f"{r['count_acc']:>7.0%}" if r else "      -")
        print(f"K={k:<4d}  " + " ".join(cells))

    print("\nmean event-log length (tokens/episode)")
    print("     mf " + " ".join(f"{m:>7d}" for m in mfs))
    for k in ks:
        cells = []
        for m in mfs:
            r = latest.get((k, m))
            cells.append(f"{r['mean_log_len']:>7.1f}" if r else "      -")
        print(f"K={k:<4d}  " + " ".join(cells))

    print("\nper-K properties (independent of min_frequency)")
    print(f"{'K':>5} {'live':>5} {'chg':>7} {'run':>6} {'bP':>6} {'bR':>6} {'MI':>7}")
    for k in ks:
        r = next(v for (kk, _), v in latest.items() if kk == k)
        print(
            f"{k:>5} {r['live_codes']:>5} {r['change_rate']:>6.1%} "
            f"{r['mean_code_run']:>6.1f} {r['boundary_precision']:>6.3f} "
            f"{r['boundary_recall']:>6.3f} {r['label_mi_frac']:>6.1%}"
        )

    best = max(latest.values(), key=lambda r: r["count_acc"])
    print(
        f"\nbest cell: K={best['clusters']} mf={best['min_frequency']} "
        f"-> {best['count_acc']:.0%} (over {best['count_over']}, under {best['count_under']})"
    )
    print("Treat one strong cell as a hypothesis, not a result; check its neighbours.")


if __name__ == "__main__":
    main()
