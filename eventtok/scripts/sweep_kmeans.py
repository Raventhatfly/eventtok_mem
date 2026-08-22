"""One cell of the (K, min_frequency) grid; appends a row to a shared JSONL.

    python -m eventtok.scripts.sweep_kmeans --task SwingXtimes --clusters 32 --min-frequency 15

K and min_frequency interact — K sets how finely the action space is quantised,
min_frequency sets how aggressively BPE merges the resulting runs — so the grid
is reported whole rather than as a best cell. Reading one lucky combination as a
finding is the obvious way to fool yourself here.

CPU only. k-means, BPE and every metric in this file are CPU work; asking for a
GPU would waste more than the requeue partition saves.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from .. import paths
from ..bpe import build_vocab as bpe
from ..data.index import RoboMMEIndex
from ..data.meta import TaskMeta
from ..eval.repeatability import full_report
from ..models.kmeans import KMeansTokenizer
from .build_events import runs_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="SwingXtimes")
    ap.add_argument("--clusters", type=int, required=True)
    ap.add_argument("--min-frequency", type=int, required=True)
    ap.add_argument("--min-span", type=int, default=3)
    ap.add_argument("--vocab-size", type=int, default=128)
    ap.add_argument("--max-token-length", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--out", default=None, help="JSONL to append to")
    args = ap.parse_args()

    paths.check_root()
    index = RoboMMEIndex()
    eps = index.by_task(args.task)
    if args.episodes:
        eps = eps[: args.episodes]
    meta = TaskMeta(args.task)

    t0 = time.time()
    km = KMeansTokenizer(args.clusters).fit(meta, eps)
    streams = {ep.epis_idx: km.stream_for_episode(meta, ep) for ep in eps}

    rep = full_report(streams, {e.epis_idx: e for e in eps}, meta)

    runs = {e: runs_of(c, args.min_span) for e, c in streams.items()}
    vocab = bpe.train(
        list(runs.values()),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        max_token_length=args.max_token_length,
    )
    bpe.assert_no_self_merges(vocab)

    exact = over = under = 0
    log_lens = []
    for ep in eps:
        tokens = vocab.encode_span(runs[ep.epis_idx])
        log_lens.append(len(tokens))
        counts: dict[int, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        top = max(counts.values()) if counts else 0
        if ep.count is not None:
            exact += top == ep.count
            over += top > ep.count
            under += top < ep.count

    row = {
        "task": args.task,
        "clusters": args.clusters,
        "min_frequency": args.min_frequency,
        "min_span": args.min_span,
        "episodes": len(eps),
        "live_codes": len({c for s in streams.values() for c in s}),
        "change_rate": rep["within_event_change_rate"],
        "mean_code_run": rep["mean_code_run"],
        "boundary_precision": rep["boundary_precision"],
        "boundary_recall": rep["boundary_recall"],
        "boundary_f1": rep["boundary_f1"],
        "label_mi_frac": rep["label_mi_frac"],
        "ngram_exact_CIRCULAR": rep["ngram_exact_CIRCULAR"],
        "merges": len(vocab.merges),
        "vocab": vocab.size,
        "mean_log_len": sum(log_lens) / max(len(log_lens), 1),
        "count_exact": exact,
        "count_over": over,
        "count_under": under,
        "count_acc": exact / max(len(eps), 1),
        "secs": round(time.time() - t0, 1),
    }
    print(json.dumps(row))

    out = args.out or str(paths.CACHE_ROOT / "eval" / f"sweep_kmeans_{args.task}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "a") as fh:            # append: array tasks share one file
        fh.write(json.dumps(row) + "\n")
    print(
        f"K={args.clusters:4d} mf={args.min_frequency:3d} -> "
        f"count {exact}/{len(eps)} ({row['count_acc']:.0%})  "
        f"log {row['mean_log_len']:.1f} tok  "
        f"bP={row['boundary_precision']:.3f} MI={row['label_mi_frac']:.1%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
