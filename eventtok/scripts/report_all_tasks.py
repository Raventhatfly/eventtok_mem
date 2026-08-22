"""Aggregate the 16 per-task rows into one table, and say what predicts the scores.

    python -m eventtok.scripts.report_all_tasks

Accuracies are not comparable across tasks -- the majority-label rate runs from ~28%
to over 50% -- so the ranking column is **gain over that task's own majority
baseline**, and the raw accuracy is printed beside it rather than instead of it.

The analysis this is for: the pipeline was developed on SwingXtimes, ButtonUnmask and
PickXtimes, and the hypothesis is that it works where events are motions and fails
where an event is defined by something off the trajectory -- a demo video to imitate,
an occluded object. The Video* tasks are the natural test of that, since they are the
ones carrying a pre-execution prefix.
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np

from .. import paths

DEV_TASKS = {"SwingXtimes", "ButtonUnmask", "PickXtimes"}


def _corr(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or x[ok].std() < 1e-9 or y[ok].std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default=None)
    args = ap.parse_args()

    pattern = args.pattern or str(paths.CACHE_ROOT / "eval" / "alltasks_*.json")
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            rows.append(json.load(fh))
    if not rows:
        raise SystemExit(f"no results matching {pattern}")

    for r in rows:
        base = r.get("label_majority_heldout", r["majority_label_rate"])
        r["_base"] = base
        for cond in ("action", "vision", "action+vision"):
            key = f"label_acc_{cond}"
            r[f"gain_{cond}"] = (r[key] - base) if key in r else float("nan")
        r["_best"] = max(
            (r.get(f"label_acc_{c}", float("nan")) for c in
             ("action", "vision", "action+vision")),
            default=float("nan"),
        )

    rows.sort(key=lambda r: -(r["gain_action+vision"] if np.isfinite(r["gain_action+vision"])
                              else r["gain_action"]))

    w = max(len(r["task"]) for r in rows)
    print(f"\n{'task':<{w}} {'ev/ep':>6} {'lbls':>5} {'maj':>6} │ "
          f"{'action':>14} {'vision':>14} {'act+vis':>14} │ {'bpeF1':>6} {'oseg':>6}")
    print("─" * (w + 84))
    for r in rows:
        def cell(c):
            k = f"label_acc_{c}"
            if k not in r:
                return f"{'-':>14}"
            return f"{r[k]:>6.1%} ({r[f'gain_{c}']:+5.1%})"
        tag = "*" if r["task"] in DEV_TASKS else " "
        print(
            f"{r['task']:<{w}}{tag}{r['events_per_episode']:>5.1f} "
            f"{r['distinct_labels']:>5} {r['_base']:>6.1%} │ "
            f"{cell('action')} {cell('vision')} {cell('action+vision')} │ "
            f"{r.get('bpe_f1', float('nan')):>6.3f} "
            f"{r.get('over_segmentation_bpe', float('nan')):>5.1f}x"
        )
    print(f"\n  * = a task the pipeline was developed on. {len(rows)} tasks reported.")

    dev = [r for r in rows if r["task"] in DEV_TASKS]
    new = [r for r in rows if r["task"] not in DEV_TASKS]
    if dev and new:
        for label, group in (("development tasks", dev), ("unseen tasks", new)):
            g = [r["gain_action+vision"] if np.isfinite(r["gain_action+vision"])
                 else r["gain_action"] for r in group]
            f1 = [r.get("bpe_f1", float("nan")) for r in group]
            print(
                f"  {label:<20} n={len(group):<3} median gain over majority "
                f"{np.nanmedian(g):+.1%}   median BPE F1 {np.nanmedian(f1):.3f}"
            )
        print(
            "  A large gap here would mean the numbers reported so far do not "
            "describe the method,\n  only the tasks it was tuned on."
        )

    vid = [r for r in rows if r["task"].startswith("Video")]
    non = [r for r in rows if not r["task"].startswith("Video")]
    if vid and non:
        print()
        for label, group in (("Video* (demo prefix)", vid), ("other", non)):
            g = [r["gain_action+vision"] if np.isfinite(r["gain_action+vision"])
                 else r["gain_action"] for r in group]
            print(f"  {label:<22} n={len(group):<3} median gain {np.nanmedian(g):+.1%}")

    print("\n  what predicts the gain (Pearson r over tasks):")
    target = [r["gain_action+vision"] if np.isfinite(r["gain_action+vision"])
              else r["gain_action"] for r in rows]
    for name, key in [
        ("events per episode", "events_per_episode"),
        ("distinct labels", "distinct_labels"),
        ("frames per episode", "frames_per_episode"),
        ("majority rate", "_base"),
    ]:
        print(f"    {name:<22} r = {_corr([r[key] for r in rows], target):+.2f}")

    finite = [r for r in rows if np.isfinite(r["gain_action"]) and np.isfinite(r["gain_vision"])]
    if finite:
        d = [r["label_acc_vision"] - r["label_acc_action"] for r in finite]
        wins = sum(1 for x in d if x > 0)
        print(
            f"\n  vision minus action, per task: median {np.median(d):+.1%}, "
            f"vision ahead on {wins}/{len(d)} tasks"
        )
        dm = [r["label_acc_action+vision"] - max(r["label_acc_action"], r["label_acc_vision"])
              for r in finite if "label_acc_action+vision" in r]
        if dm:
            print(
                f"  combining beats the better single modality by a median of "
                f"{np.median(dm):+.1%}, on {sum(1 for x in dm if x > 0)}/{len(dm)} tasks"
            )


if __name__ == "__main__":
    main()
