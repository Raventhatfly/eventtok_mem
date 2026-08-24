"""Does the BPE stage produce event-granularity segmentation?

    python -m eventtok.scripts.eval_bpe_boundaries --task SwingXtimes

Sweeps (k, min_frequency) and reports, for each cell, the boundary score of the BPE
token stream **and** of the run symbols it was built from. The run-symbol row is the
thing BPE has to beat; without it a mediocre BPE score is unreadable, since
run-length encoding alone already removes most boundaries.

Fitting note: the vocabulary is trained on train episodes only and applied to
held-out ones. BPE learns which code sequences recur, so training it on all episodes
and scoring the same episodes would let it memorise the very spans being scored.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .. import paths
from ..bpe import build_vocab as bpe
from ..data.index import RoboMMEIndex
from ..data.meta import TaskMeta
from ..eval.bpe_boundaries import report, runs_with_spans
from ..models.kmeans import KMeansTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="SwingXtimes")
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--min-frequencies", type=int, nargs="+", default=[5, 10, 25, 50])
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--max-token-length", type=int, default=4)
    ap.add_argument("--min-span", type=int, default=3)
    ap.add_argument("--tolerance", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths.check_root()
    index = RoboMMEIndex()
    eps = index.by_task(args.task)
    meta = TaskMeta(args.task)

    order = np.random.default_rng(args.seed).permutation(len(eps))
    cut = len(eps) // 2
    train_eps = [eps[i] for i in order[:cut]]
    test_eps = [eps[i] for i in order[cut:]]
    print(
        f"{args.task}: {len(train_eps)} train / {len(test_eps)} held-out episodes, "
        f"tolerance {args.tolerance} transitions",
        flush=True,
    )

    rows = []
    for k in args.ks:
        km = KMeansTokenizer(k, seed=args.seed).fit(meta, train_eps)
        streams = {
            ep.epis_idx: km.stream_for_episode(meta, ep) for ep in eps
        }
        # BPE corpus: run-symbol sequences from TRAIN episodes only.
        corpus = [
            [r.symbol for r in runs_with_spans(streams[ep.epis_idx], args.min_span)]
            for ep in train_eps
        ]
        for mf in args.min_frequencies:
            vocab = bpe.train(
                corpus,
                vocab_size=args.vocab_size,
                min_frequency=mf,
                max_token_length=args.max_token_length,
            )
            bpe.assert_no_self_merges(vocab)
            res = report(
                {ep.epis_idx: streams[ep.epis_idx] for ep in test_eps},
                {ep.epis_idx: ep for ep in test_eps},
                meta,
                vocab,
                min_span=args.min_span,
                tolerance=args.tolerance,
            )
            res.update({"k": k, "min_frequency": mf})
            rows.append(res)
            print(
                f"  k={k:4d} mf={mf:3d}  merges {res['merges']:3d}  "
                f"tokens/ep {res['bpe_tokens_per_episode']:6.1f} "
                f"(runs {res['run_symbols_per_episode']:5.1f}, "
                f"true {res['true_events_per_episode']:4.1f})  "
                f"runs F1 {res['runs_f1']:.3f} -> bpe F1 {res['bpe_f1']:.3f} "
                f"({res['bpe_f1_gain']:+.3f})  "
                f"bpe P {res['bpe_precision']:.3f} R {res['bpe_recall']:.3f}  "
                f"over-seg {res['over_segmentation_bpe']:.1f}x",
                flush=True,
            )

    best = max(rows, key=lambda r: r["bpe_f1"])
    print(
        f"\n  best BPE F1 {best['bpe_f1']:.3f} at k={best['k']} mf={best['min_frequency']} "
        f"(run-symbol F1 there {best['runs_f1']:.3f}, gain {best['bpe_f1_gain']:+.3f})"
    )
    gains = [r["bpe_f1_gain"] for r in rows]
    print(
        f"  BPE F1 gain over run symbols across all {len(rows)} cells: "
        f"min {min(gains):+.3f} median {float(np.median(gains)):+.3f} max {max(gains):+.3f}"
    )
    if max(gains) <= 0:
        print(
            "  BPE does not improve boundaries in ANY cell -- the merging stage is "
            "not doing the job the design assigns it"
        )
    closest = min(rows, key=lambda r: abs(r["over_segmentation_bpe"] - 1.0))
    print(
        f"  closest to event granularity: k={closest['k']} mf={closest['min_frequency']} "
        f"at {closest['over_segmentation_bpe']:.1f}x "
        f"({closest['bpe_tokens_per_episode']:.1f} tokens vs "
        f"{closest['true_events_per_episode']:.1f} true events)"
    )

    out = args.out or str(paths.CACHE_ROOT / "eval" / f"bpe_boundaries_{args.task}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"task": args.task, "rows": rows}, fh, indent=2)
    print("\n  wrote", out)


if __name__ == "__main__":
    main()
