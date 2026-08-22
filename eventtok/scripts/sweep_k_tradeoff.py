"""Label accuracy and counting, on the same code streams, across the cluster count.

    python -m eventtok.scripts.sweep_k_tradeoff --task SwingXtimes

Why this exists. The encoder sweep showed label accuracy climbing monotonically with
k -- action-only goes 69.6% -> 89.8% over k=16..128 on SwingXtimes. Taken alone that
says pick the largest k, which cannot be right: the whole design needs a code stream
where a repeated event produces a repeated code, and finer clusters split repetitions
apart. "Pin k" is only a well-posed instruction if k is chosen against a metric that
penalises over-fragmentation, and label accuracy does not.

So both are measured here on identical streams:

  * label accuracy -- can the code identify which event this is (with its majority
    baseline, which is not optional);
  * counting -- does a pattern selected on train episodes, using train N only, occur
    exactly N times on held-out episodes (with its shuffled-label baseline, ~37%).

The counting side follows eventtok.eval.counting, which exists because three earlier
attempts at this were circular or baseline-free. The pattern is chosen without ever
consulting held-out N.

If the two metrics disagree about k -- and the point of running it is that they
might -- then label accuracy is the wrong criterion for choosing a codebook, and any
downstream decision made on it needs revisiting.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .. import paths
from ..data import dino, repack
from ..data.index import RoboMMEIndex
from ..data.meta import TaskMeta
from ..eval import counting as cnt
from ..eval.repeatability import label_accuracy, label_mutual_information
from .compare_modalities import Block, action_matrix, kmeans_fit_predict, vision_matrix


def stream_stats(streams: dict[int, list[int]], min_span: int) -> dict:
    """Fragmentation, which is what a large k costs and label accuracy hides."""
    runs = {e: cnt.runs_of(c, min_span) for e, c in streams.items()}
    lens = [len(r) for r in runs.values()]
    changes = [
        sum(1 for i in range(1, len(c)) if c[i] != c[i - 1]) / max(len(c) - 1, 1)
        for c in streams.values()
    ]
    return {
        "live_codes": len({c for s in streams.values() for c in s}),
        "mean_run_symbols": float(np.mean(lens)),
        "mean_change_rate": float(np.mean(changes)),
        "runs": runs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="SwingXtimes")
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--inputs", default="action", choices=["action", "action+vision"])
    ap.add_argument("--encoder", default="siglip")
    ap.add_argument("--camera", default="image")
    ap.add_argument("--scale", default="2x2")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--min-span", type=int, default=3)
    ap.add_argument("--max-pattern-len", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths.check_root()
    index = RoboMMEIndex()
    eps = index.by_task(args.task)
    counts = {ep.epis_idx: ep.count for ep in eps}
    if any(v is None for v in counts.values()):
        raise SystemExit(
            f"{args.task} has no per-episode N; counting needs a *Xtimes task"
        )
    meta = TaskMeta(args.task)

    order = np.random.default_rng(args.seed).permutation(len(eps))
    cut = len(eps) // 2
    train_ids = [eps[i].epis_idx for i in order[:cut]]
    test_ids = [eps[i].epis_idx for i in order[cut:]]
    train_set = set(train_ids)

    rows, epis_of_row, frame_of_row = [], [], []
    for ep in eps:
        lo, hi = meta.rows(ep.epis_idx)
        rows.extend(range(lo, hi))
        epis_of_row.extend([ep.epis_idx] * (hi - lo))
        frame_of_row.extend(range(hi - lo))
    rows = np.asarray(rows)
    epis_of_row = np.asarray(epis_of_row)
    frame_of_row = np.asarray(frame_of_row)
    train_mask = np.array([e in train_set for e in epis_of_row])

    blocks_raw = {"action": action_matrix(meta, rows)}
    if args.inputs == "action+vision":
        if args.encoder == "siglip" and args.camera == "image":
            getter = repack.EpisodeFeatures(args.task, args.scale)
        else:
            getter = dino.EpisodeDinoFeatures(
                args.task, args.encoder, args.scale, args.camera
            )
        blocks_raw["vision"] = vision_matrix(
            getter, epis_of_row, frame_of_row, args.horizon, "both"
        )

    fitted = {
        name: Block(name, None if name == "action" else args.pca).fit(
            X[train_mask], seed=args.seed
        )
        for name, X in blocks_raw.items()
    }
    X = np.concatenate([fitted[n].transform(blocks_raw[n]) for n in blocks_raw], axis=1)
    print(
        f"{args.task}: {len(eps)} episodes, {len(rows)} transitions, "
        f"inputs={args.inputs} dims={X.shape[1]} "
        f"train {len(train_ids)} / test {len(test_ids)}",
        flush=True,
    )

    results = []
    for k in args.ks:
        codes, _ = kmeans_fit_predict(X[train_mask], X, k, args.seed)
        streams = {}
        for ep in eps:
            lo, hi = meta.rows(ep.epis_idx)
            streams[ep.epis_idx] = codes[lo:hi].tolist()

        all_codes, all_labels, all_train = [], [], []
        for ep in eps:
            c, l = label_mutual_information(streams[ep.epis_idx], ep, meta)
            all_codes.extend(c)
            all_labels.extend(l)
            all_train.extend([ep.epis_idx in train_set] * len(c))
        acc, majority = label_accuracy(all_codes, all_labels, all_train)

        st = stream_stats(streams, args.min_span)
        runs = st.pop("runs")

        pattern = cnt.select_pattern(
            runs, counts, train_ids, max_len=args.max_pattern_len
        )
        if pattern is None:
            row = {
                "k": k, "label_accuracy": acc, "label_majority": majority,
                "count_accuracy": None, "count_baseline": None, **st,
            }
            results.append(row)
            print(f"  k={k:4d}  label {acc:6.1%}  counting: no usable pattern", flush=True)
            continue

        ev = cnt.evaluate(runs, counts, pattern, test_ids)
        base_mean, base_sd = cnt.shuffled_baseline(runs, counts, pattern, test_ids)
        row = {
            "k": k,
            "label_accuracy": acc,
            "label_majority": majority,
            "count_accuracy": ev["accuracy"],
            "count_mae": ev["mae"],
            "count_baseline": base_mean,
            "count_baseline_sd": base_sd,
            "pattern": list(pattern.gram),
            "pattern_len": pattern.length,
            "pattern_train_accuracy": pattern.train_accuracy,
            **st,
        }
        results.append(row)
        print(
            f"  k={k:4d}  label {acc:6.1%} (maj {majority:.1%})   "
            f"count {ev['accuracy']:6.1%} (chance {base_mean:.1%}+-{base_sd:.1%})   "
            f"live {st['live_codes']:4d}  run-symbols/ep {st['mean_run_symbols']:6.1f}  "
            f"pattern {tuple(pattern.gram)}",
            flush=True,
        )

    print("\n  where the two metrics point:")
    got_counts = [r for r in results if r["count_accuracy"] is not None]
    if got_counts:
        best_label = max(results, key=lambda r: r["label_accuracy"])
        best_count = max(got_counts, key=lambda r: r["count_accuracy"])
        print(f"    best label accuracy   k={best_label['k']:<4} {best_label['label_accuracy']:.1%}")
        print(f"    best counting         k={best_count['k']:<4} {best_count['count_accuracy']:.1%}")
        if best_label["k"] != best_count["k"]:
            print(
                "    THEY DISAGREE -- label accuracy is not a valid criterion for "
                "choosing the codebook size on its own"
            )
        else:
            print("    same k; label accuracy and counting agree here")

    out = args.out or str(
        paths.CACHE_ROOT / "eval" / f"ktradeoff_{args.task}_{args.inputs.replace('+','_')}.json"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"task": args.task, "inputs": args.inputs, "results": results}, fh, indent=2)
    print("\n  wrote", out)


if __name__ == "__main__":
    main()
