"""End-to-end: checkpoint -> code streams -> BPE vocabulary -> event logs.

    python -m eventtok.scripts.build_events --ckpt ... --task SwingXtimes

Stages, and what each is responsible for:

  1. tokenize   every transition -> one code. High recall, over-segmented ~4-8x.
                A code change is a *candidate* boundary, not an event boundary.
  2. segment    split the stream at code changes (min_span guards jitter). These
                are the spans BPE trains on -- produced by the same online rule
                used at test time, so the merges match what will be seen.
  3. BPE        merge frequent adjacent pairs into event-sized units. This is the
                selection stage that collapses over-segmentation.
  4. log        append-only event log per episode, plus counts.

Reports boundary alignment and label MI alongside the counts, because
within-event stability alone cannot tell segmentation from a constant code.
"""

from __future__ import annotations

import argparse
import json

import torch

from .. import paths
from ..bpe import build_vocab as bpe
from ..bpe.streaming import CodeChangeBoundary, StreamingTokenizer
from ..data.index import RoboMMEIndex
from ..data.meta import TaskMeta
from ..data.robomme import TransitionDataset
from ..eval.repeatability import full_report
from .tokenize_episodes import load_model, stream_for_episode


def runs_of(codes: list[int], min_span: int) -> list[int]:
    """Collapse a code stream into its sequence of run symbols.

    Runs are the *candidate* units: the code stream over-segments events 4-8x, so
    a run is finer than an event. BPE then merges runs into event-sized tokens —
    that is the selection stage.

    Running BPE *inside* runs instead is self-defeating: a run is by definition a
    constant stretch, so every adjacent pair within it is a self-pair, which the
    no-self-merge guard forbids. That produced 0 merges and an "event log" that was
    just the raw stream (429 tokens for a 7-event episode).
    """
    st = StreamingTokenizer(lambda s: [int(s[0])], CodeChangeBoundary(min_span))
    for c in codes:
        st.push(c)
    st.finish()
    return list(st.log.tokens)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="neural tokenizer checkpoint; omit when --kmeans is set")
    ap.add_argument("--kmeans", type=int, default=None,
                    help="use k-means with this many clusters instead of a network")
    ap.add_argument("--task", default="SwingXtimes")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--min-span", type=int, default=2)
    ap.add_argument("--vocab-size", type=int, default=64)
    ap.add_argument("--min-frequency", type=int, default=10)
    ap.add_argument("--max-token-length", type=int, default=20)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    paths.check_root()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    index = RoboMMEIndex()
    eps = index.by_task(args.task)
    if args.episodes:
        eps = eps[: args.episodes]
    meta = TaskMeta(args.task)

    # 1. tokenize
    streams: dict[int, list[int]] = {}
    if args.kmeans:
        from ..models.kmeans import KMeansTokenizer

        km = KMeansTokenizer(args.kmeans).fit(meta, eps)
        for ep in eps:
            streams[ep.epis_idx] = km.stream_for_episode(meta, ep)
        km.save(paths.CACHE_ROOT / "ckpt" / f"kmeans_{args.task}_{args.kmeans}.npz")
        tag = f"kmeans{args.kmeans}"
    else:
        if not args.ckpt:
            ap.error("pass --ckpt or --kmeans")
        model, cfg = load_model(args.ckpt, device)
        ds = TransitionDataset(
            args.task, k=cfg["k"], scale=cfg["scale"], episodes=eps, index=index
        )
        for ep in eps:
            codes, _, _ = stream_for_episode(model, ds, ep.epis_idx, device)
            streams[ep.epis_idx] = codes
        tag = "neural"
    print(f"tokenized {len(eps)} episodes, {sum(len(v) for v in streams.values())} transitions")

    rep = full_report(streams, {e.epis_idx: e for e in eps}, meta)
    print(
        f"  code stream: change rate {rep['within_event_change_rate']:.1%}, "
        f"boundary P={rep['boundary_precision']:.3f} R={rep['boundary_recall']:.3f} "
        f"F1={rep['boundary_f1']:.3f}, label MI {rep['label_mi_frac']:.1%}"
    )

    # 2. runs (candidate units)  ->  3. BPE over run sequences
    runs = {e: runs_of(c, args.min_span) for e, c in streams.items()}
    corpus = list(runs.values())     # one sequence per episode; no cross-episode merges
    print(
        f"  {sum(map(len, corpus))} runs over {len(corpus)} episodes "
        f"(mean {sum(map(len, corpus))/max(len(corpus),1):.1f} runs/episode)"
    )
    vocab = bpe.train(
        corpus,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        max_token_length=args.max_token_length,
    )
    bpe.assert_no_self_merges(vocab)
    print(f"  BPE: {len(vocab.merges)} merges, {vocab.size} tokens (guards pass)")

    # 4. event logs
    logs: dict[int, dict] = {}
    exact = over = under = 0
    for ep in eps:
        from ..bpe.streaming import EventLog

        tokens = vocab.encode_span(runs[ep.epis_idx])
        st_log = EventLog(tokens=list(tokens))
        counts = st_log.counts()
        st = type("obj", (), {"log": st_log})()
        top = max(counts.values()) if counts else 0
        if ep.count is not None:
            exact += top == ep.count
            over += top > ep.count
            under += top < ep.count
        logs[ep.epis_idx] = {
            "tokens": st.log.tokens,
            "rle": st.log.run_length_encode(),
            "counts": {str(k): v for k, v in counts.items()},
            "N": ep.count,
        }

    mean_len = sum(len(v["tokens"]) for v in logs.values()) / max(len(logs), 1)
    print(f"  event logs: mean {mean_len:.1f} tokens/episode")
    print(f"  most-frequent-token count vs N: exact {exact}/{len(eps)} over {over} under {under}")

    out = paths.CACHE_ROOT / "events" / f"{args.task}_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"logs": {str(k): v for k, v in logs.items()}, "report": {
            k: v for k, v in rep.items() if k != "rows"}}, fh, indent=2)
    vocab.save(out.with_name(f"{args.task}_{tag}_vocab.json"))
    print("wrote", out)


if __name__ == "__main__":
    main()
