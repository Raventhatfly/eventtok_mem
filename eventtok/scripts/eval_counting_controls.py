"""Tokenizer counting vs. baselines that need no tokenizer. Same split, same protocol.

    python -m eventtok.scripts.eval_counting_controls --task PickXtimes

The 96% extrapolation result is only a finding if something trivial does not match it.
Every predictor here is fitted on the N in {1,2,3} episodes and applied unchanged to
held-out in-distribution episodes and to the N in {4,5} episodes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

import numpy as np

from .. import paths
from ..data.index import RoboMMEIndex
from ..data.meta import TaskMeta
from ..eval import counting as cnt
from ..eval.trivial_baselines import fit_predictors
from ..models.kmeans import KMeansTokenizer


def score(pred: dict[int, int], counts: dict[int, int], ids, seed: int = 0) -> dict:
    truth = np.array([counts[e] for e in ids], float)
    got = np.array([pred[e] for e in ids], float)
    rng = random.Random(seed)
    labels = [counts[e] for e in ids]
    shuf = []
    for _ in range(400):
        perm = labels[:]
        rng.shuffle(perm)
        shuf.append(sum(1 for p, t in zip(got, perm) if p == t) / len(ids))
    return {
        "n": len(ids),
        "accuracy": float((got == truth).mean()),
        "mae": float(np.abs(got - truth).mean()),
        "shuffled": float(np.mean(shuf)),
        "hist": dict(sorted(Counter(int(v) for v in got).items())),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickXtimes")
    ap.add_argument("--train-n", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--test-n", type=int, nargs="*", default=[4, 5],
                    help="empty for no OOD split; the in-distribution column still applies")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--min-span", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths.check_root()
    index = RoboMMEIndex()
    meta = TaskMeta(args.task)
    in_dist = index.by_count(args.task, args.train_n)
    ood = index.by_count(args.task, args.test_n) if args.test_n else []
    counts = {e.epis_idx: e.count for e in in_dist + ood}

    # A single-valued evaluation set makes accuracy meaningless: shuffling identical
    # labels leaves them identical, so chance is 100% and a constant predictor is
    # perfect. Holding out N=3 on SwingXtimes does exactly this, and it produced a
    # "+67.4% over the best control" that meant nothing.
    if ood and len({counts[e.epis_idx] for e in ood}) < 2:
        raise SystemExit(
            f"OOD set has a single N value ({sorted({counts[e.epis_idx] for e in ood})}); "
            f"accuracy against it is degenerate. {args.task} tops out at "
            f"N={max(counts.values())}, so it cannot support an extrapolation split -- "
            f"use PickXtimes, or pass --test-n '' for the in-distribution comparison."
        )

    order = np.random.default_rng(args.seed).permutation(len(in_dist))
    cut = len(in_dist) // 2
    fit_eps = [in_dist[i] for i in order[:cut]]
    id_eps = [in_dist[i] for i in order[cut:]]
    fit_ids = [e.epis_idx for e in fit_eps]
    id_ids = [e.epis_idx for e in id_eps]
    ood_ids = [e.epis_idx for e in ood]

    print(
        f"{args.task}: fit {len(fit_ids)} (N in {args.train_n}), "
        f"held-out in-dist {len(id_ids)}, OOD {len(ood_ids)} (N in {args.test_n})",
        flush=True,
    )

    preds: dict[str, dict[int, int]] = {}

    # --- the tokenizer -----------------------------------------------------
    km = KMeansTokenizer(args.k, seed=args.seed).fit(meta, fit_eps)
    streams = {e.epis_idx: km.stream_for_episode(meta, e) for e in in_dist + ood}
    runs = {e: cnt.runs_of(s, args.min_span) for e, s in streams.items()}
    pattern = cnt.select_pattern(runs, counts, fit_ids, max_len=6)
    from collections import Counter as C
    def grams(seq, L):
        return [tuple(seq[i:i+L]) for i in range(len(seq)-L+1)]
    preds[f"tokenizer k={args.k} {tuple(pattern.gram)}"] = {
        e: C(grams(runs[e], pattern.length))[pattern.gram] for e in id_ids + ood_ids
    }

    # --- the controls ------------------------------------------------------
    trivial = fit_predictors(meta, fit_eps, counts)
    for name, spec in trivial.items():
        by_ep = {e.epis_idx: e for e in in_dist + ood}
        preds[name] = {i: spec["predict"](by_ep[i]) for i in id_ids + ood_ids}
        if spec["param"] is not None:
            print(f"  {name}: parameter {spec['param']:.4g} "
                  f"(fit accuracy {spec['fit_acc']:.0%})", flush=True)

    rows = []
    width = max(len(n) for n in preds)
    print(f"\n  {'predictor':<{width}}  {'in-dist acc':>12}  {'OOD acc':>9}  "
          f"{'OOD chance':>10}  {'OOD MAE':>8}   OOD predictions")
    empty = {"n": 0, "accuracy": float("nan"), "mae": float("nan"),
             "shuffled": float("nan"), "hist": {}}
    for name, pred in preds.items():
        r_id = score(pred, counts, id_ids, args.seed)
        r_ood = score(pred, counts, ood_ids, args.seed) if ood_ids else dict(empty)
        rows.append({"predictor": name, "in_distribution": r_id,
                     "out_of_distribution": r_ood})
        print(
            f"  {name:<{width}}  {r_id['accuracy']:>11.1%}  {r_ood['accuracy']:>8.1%}  "
            f"{r_ood['shuffled']:>9.1%}  {r_ood['mae']:>7.2f}   {r_ood['hist']}",
            flush=True,
        )

    split = "out_of_distribution" if ood_ids else "in_distribution"
    tok = rows[0]
    best_ctrl = max(rows[1:], key=lambda r: r[split]["accuracy"])
    gap = tok[split]["accuracy"] - best_ctrl[split]["accuracy"]
    print(
        f"\n  tokenizer minus best control on {split}: {gap:+.1%} "
        f"(best control: {best_ctrl['predictor']})"
    )
    if gap <= 0:
        print(
            "  A predictor needing no tokenizer matches or beats it. The counting\n"
            "  benchmark does not distinguish them, so it cannot support the claim."
        )
    else:
        print("  The tokenizer beats every control that needs no representation.")

    out = args.out or str(paths.CACHE_ROOT / "eval" / f"controls_{args.task}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"task": args.task, "k": args.k, "rows": rows}, fh, indent=2)
    print("\n  wrote", out)


if __name__ == "__main__":
    main()
