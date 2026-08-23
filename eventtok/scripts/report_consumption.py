"""Consolidate the consumption runs into one table and one verdict.

    python -m eventtok.scripts.report_consumption

Percentages are relative to the ``none`` condition on the same task, because absolute
L1 is not comparable across tasks -- action scales and episode structure differ.

The verdict line keys on ``log`` vs ``wrong``, expressed as a share of the total
benefit of having any memory at all. That framing is the useful one: a log that helps
by 10% while a wrong log helps by 9% is a policy detecting that *something* is attached,
not reading it.
"""

from __future__ import annotations

import glob
import json

import numpy as np

from .. import paths

ORDER = ["none", "log", "wrong", "shuffled", "count"]


def main() -> None:
    rows = []
    for path in sorted(glob.glob(str(paths.CACHE_ROOT / "eval" / "consumption_*.json"))):
        with open(path) as fh:
            rows.append(json.load(fh))
    if not rows:
        raise SystemExit("no consumption results yet")

    w = max(len(r["task"]) for r in rows)
    print(f"\n  held-out action L1, relative to `none` on the same task\n")
    print(f"  {'task':<{w}} " + "".join(f"{c:>12}" for c in ORDER) + f"{'ref(mean)':>11}")
    print("  " + "─" * (w + 12 * len(ORDER) + 11))
    for r in rows:
        res = r["results"]
        base = res.get("none", {}).get("l1")
        cells = ""
        for c in ORDER:
            if c not in res:
                cells += f"{'-':>12}"
            elif c == "none":
                cells += f"{res[c]['l1']:>12.4f}"
            else:
                cells += f"{(res[c]['l1'] / base - 1) * 100:>+11.1f}%"
        print(f"  {r['task']:<{w}} {cells}{r['reference_mean_L1']:>11.4f}")

    print("\n  contrasts, as a share of the total benefit of having memory")
    print(f"  {'task':<{w}} {'log helps':>11} {'content':>10} {'order':>9} {'vs count':>10}")
    shares, orders, counts = [], [], []
    for r in rows:
        res = r["results"]
        if not {"none", "log"} <= set(res):
            continue
        base, lg = res["none"]["l1"], res["log"]["l1"]
        benefit = base - lg
        helps = benefit / base
        # Shares are undefined when memory does not help -- the denominator is zero or
        # negative and the ratio explodes rather than degrading. Tasks where the log
        # hurts are reported as such instead of with a fabricated percentage.
        usable = benefit > 1e-4

        def share(c):
            if c not in res or not usable:
                return float("nan")
            return (res[c]["l1"] - lg) / benefit

        s_wrong, s_shuf, s_count = share("wrong"), share("shuffled"), share("count")
        shares.append(s_wrong); orders.append(s_shuf); counts.append(s_count)
        def fmt(x):
            return f"{x:>9.0%}" if np.isfinite(x) else f"{'n/a':>9}"
        note = "" if usable else "   (log does not help; shares undefined)"
        print(f"  {r['task']:<{w}} {helps:>10.1%} {fmt(s_wrong)} {fmt(s_shuf)} "
              f"{fmt(s_count)}{note}")

    print("\n    content = how much of the benefit disappears with a WRONG log.")
    print("              100% means the log is read; 0% means only its presence matters.")
    print("    order   = how much disappears when the tokens are shuffled.")
    print("    vs count= positive means the raw log beats a plain count vector.")

    if shares:
        m = float(np.nanmedian(shares))
        print(f"\n  median content share {m:.0%} across {len(shares)} tasks")
        if m < 0.25:
            print("  The policy is largely ignoring what the log says. Any claim built on\n"
                  "  the log's content does not survive this.")
        elif m > 0.6:
            print("  The log's content is being read, not just its presence. The wrong-log\n"
                  "  control passes, which is the precondition for every downstream claim.")
        else:
            print("  Partial: some of the benefit is content, some is mere presence.\n"
                  "  Report both numbers, never the log-vs-none figure alone.")
    if orders and not np.all(np.isnan(orders)):
        mo = float(np.nanmedian(orders))
        print(f"  median order share {mo:.0%} -- "
              + ("order matters" if mo > 0.3 else
                 "order barely matters; the multiset carries most of it"))
    if counts and not np.all(np.isnan(counts)):
        mc = float(np.nanmedian(counts))
        print(f"  median raw-log advantage over a count vector {mc:+.0%} -- "
              + ("the sequence earns its place" if mc > 0.1 else
                 "a plain count is about as good, which is a simpler system"))


if __name__ == "__main__":
    main()
