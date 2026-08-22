"""Does counting generalise to repetition counts never seen in training?

    python -m eventtok.scripts.eval_extrapolation --task PickXtimes

This is the experiment the project was built toward, and it is the one WeaveLA's
reported N=3 cannot answer. PickXtimes is the only task that can run it: N is
recoverable from the prompt and spans {1,2,3,4,5} (23/25/27/12/13 episodes), while
SwingXtimes tops out at 3.

The split. k-means and the counting pattern are fitted on a subset of N in {1,2,3}
episodes only. Nothing from N in {4,5} touches the clustering, the pattern search, or
any hyperparameter. The pattern is then counted on two disjoint evaluation sets:

    in-distribution      held-out N in {1,2,3}   -- does it work at all
    out-of-distribution  every N in {4,5}        -- does it extrapolate

Baselines, because an accuracy alone says nothing and this project has published four
numbers that turned out to be chance:

  * shuffled  -- permute the true N among the evaluation episodes. On the OOD set
    this is HIGH (~50%) because only two values occur, so OOD accuracy must clear
    50%, not 0%, to mean anything.
  * train-mode constant -- always predict the most common training N. This scores 0%
    on the OOD set by construction, and is the thing a memorising model degenerates
    to. Reported to make the failure mode legible, not as a serious competitor.

The mechanism under test is simple enough to be worth stating: on SwingXtimes,
counting worked through a *single* run symbol occurring once per repetition, with no
BPE and no vision. If that is a genuine per-repetition detector it should keep
counting past 3 without being told; if it was fitted to the training range it will
saturate. The predicted-count histogram below is what distinguishes those, so it is
printed whatever the accuracy says.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np

from .. import paths
from ..data.index import RoboMMEIndex
from ..data.meta import TaskMeta
from ..eval import counting as cnt
from ..models.kmeans import KMeansTokenizer


def evaluate_set(runs, counts, pattern, ids, label: str) -> dict:
    ev = cnt.evaluate(runs, counts, pattern, ids)
    base_mean, base_sd = cnt.shuffled_baseline(runs, counts, pattern, ids)
    pred = ev["predicted"]
    truth = np.array([counts[e] for e in ids], dtype=float)
    got = np.array([pred[e] for e in ids], dtype=float)
    corr = (
        float(np.corrcoef(got, truth)[0, 1])
        if got.std() > 1e-9 and truth.std() > 1e-9
        else float("nan")
    )
    return {
        "split": label,
        "n": len(ids),
        "accuracy": ev["accuracy"],
        "mae": ev["mae"],
        "over": ev["over"],
        "under": ev["under"],
        "shuffled": base_mean,
        "shuffled_sd": base_sd,
        "correlation": corr,
        "predicted_hist": dict(sorted(Counter(int(v) for v in got).items())),
        "true_hist": dict(sorted(Counter(int(v) for v in truth).items())),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickXtimes")
    ap.add_argument("--train-n", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--test-n", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--min-span", type=int, default=3)
    ap.add_argument("--max-pattern-len", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths.check_root()
    index = RoboMMEIndex()
    meta = TaskMeta(args.task)

    in_dist = index.by_count(args.task, args.train_n)
    ood = index.by_count(args.task, args.test_n)
    if not ood:
        raise SystemExit(f"{args.task} has no episodes with N in {args.test_n}")

    order = np.random.default_rng(args.seed).permutation(len(in_dist))
    cut = len(in_dist) // 2
    fit_eps = [in_dist[i] for i in order[:cut]]
    id_test_eps = [in_dist[i] for i in order[cut:]]

    counts = {ep.epis_idx: ep.count for ep in in_dist + ood}
    fit_ids = [ep.epis_idx for ep in fit_eps]
    id_ids = [ep.epis_idx for ep in id_test_eps]
    ood_ids = [ep.epis_idx for ep in ood]

    train_mode = Counter(counts[e] for e in fit_ids).most_common(1)[0][0]
    print(
        f"{args.task}: fit on {len(fit_ids)} episodes with N in {args.train_n}; "
        f"eval on {len(id_ids)} held-out in-distribution and {len(ood_ids)} with "
        f"N in {args.test_n}",
        flush=True,
    )
    print(
        f"  train-mode constant predictor would say N={train_mode} always -> "
        f"{sum(1 for e in ood_ids if counts[e] == train_mode) / len(ood_ids):.0%} on OOD",
        flush=True,
    )

    rows = []
    for k in args.ks:
        # Clustering sees only the fitting episodes. Fitting it on everything would
        # let the codebook adapt to the held-out counts.
        km = KMeansTokenizer(k, seed=args.seed).fit(meta, fit_eps)
        streams = {
            ep.epis_idx: km.stream_for_episode(meta, ep) for ep in in_dist + ood
        }
        runs = {e: cnt.runs_of(s, args.min_span) for e, s in streams.items()}

        pattern = cnt.select_pattern(
            runs, counts, fit_ids, max_len=args.max_pattern_len
        )
        if pattern is None:
            print(f"  k={k:4d}  no usable pattern", flush=True)
            continue

        res_id = evaluate_set(runs, counts, pattern, id_ids, "in_distribution")
        res_ood = evaluate_set(runs, counts, pattern, ood_ids, "out_of_distribution")
        row = {
            "k": k,
            "pattern": list(pattern.gram),
            "pattern_len": pattern.length,
            "pattern_fit_accuracy": pattern.train_accuracy,
            "train_mode": train_mode,
            "in_distribution": res_id,
            "out_of_distribution": res_ood,
        }
        rows.append(row)
        print(
            f"  k={k:4d} pattern {tuple(pattern.gram)}\n"
            f"        in-dist  acc {res_id['accuracy']:6.1%} "
            f"(shuffled {res_id['shuffled']:.1%}+-{res_id['shuffled_sd']:.1%})  "
            f"MAE {res_id['mae']:.2f}  r={res_id['correlation']:+.2f}\n"
            f"        OOD      acc {res_ood['accuracy']:6.1%} "
            f"(shuffled {res_ood['shuffled']:.1%}+-{res_ood['shuffled_sd']:.1%})  "
            f"MAE {res_ood['mae']:.2f}  r={res_ood['correlation']:+.2f}  "
            f"under {res_ood['under']}/{res_ood['n']}\n"
            f"        OOD predicted {res_ood['predicted_hist']} vs true "
            f"{res_ood['true_hist']}",
            flush=True,
        )

    if rows:
        best = max(rows, key=lambda r: r["out_of_distribution"]["accuracy"])
        b = best["out_of_distribution"]
        print(
            f"\n  best OOD accuracy {b['accuracy']:.1%} at k={best['k']} "
            f"against a {b['shuffled']:.1%} shuffled baseline"
        )
        # Saturation is the specific failure this experiment is looking for.
        saturating = [
            r
            for r in rows
            if max(r["out_of_distribution"]["predicted_hist"]) <= max(args.train_n)
        ]
        if saturating:
            print(
                f"  {len(saturating)}/{len(rows)} cells never predict above "
                f"N={max(args.train_n)} on OOD episodes -- the pattern saturates at "
                f"the training range rather than counting"
            )
        else:
            print(
                f"  every cell predicts above N={max(args.train_n)} somewhere on OOD "
                f"-- the pattern is not capped by the training range"
            )

    out = args.out or str(paths.CACHE_ROOT / "eval" / f"extrapolation_{args.task}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"task": args.task, "train_n": args.train_n,
                   "test_n": args.test_n, "rows": rows}, fh, indent=2)
    print("\n  wrote", out)


if __name__ == "__main__":
    main()
