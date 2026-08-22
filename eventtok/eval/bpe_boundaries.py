"""Does BPE fix the over-segmentation? The measurement that was never run.

Every boundary number in this project so far describes the **raw code stream**:
recall 0.87-1.00, precision 0.05-0.24, over-segmenting 4-10x. That was always
reported with the caveat that merging candidates into events is the BPE stage's job.
The caveat was never checked, so "the tokenization over-segments" and "the stage
meant to fix it was never measured" were being confused for each other.

The gap is concrete. Annotated events per episode: SwingXtimes 7.5, ButtonUnmask 2.5.
Run symbols per episode from k-means: 26 at k=8 rising to 76 at k=128. If BPE works,
its token count per episode lands near the first pair, and its token boundaries score
better than the run-symbol boundaries they were built from. If it does not, the
design needs a different merging stage, and knowing that is worth more than another
encoder comparison.

The comparison here is BPE tokens **against the run symbols they were built from**,
on the same episodes with the same tolerance. That isolates what BPE contributes,
which comparing against the per-transition stream would not: run-length encoding
alone already removes most of the boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..bpe.build_vocab import BPEVocab
from ..data import subgoals as sg
from ..data.index import Episode
from ..data.meta import TaskMeta
from .repeatability import BoundaryScore


@dataclass(frozen=True)
class Run:
    symbol: int
    start: int      # first transition index of the run
    end: int        # one past the last


def runs_with_spans(codes: Sequence[int], min_span: int = 3) -> list[Run]:
    """Run symbols, keeping the frame span each one covers.

    ``eval.counting.runs_of`` drops the spans; the boundary metric needs them, and
    recomputing them separately is how the two would drift apart.
    """
    out: list[Run] = []
    if not codes:
        return out
    start = 0
    for i in range(1, len(codes) + 1):
        if i == len(codes) or codes[i] != codes[start]:
            if i - start >= min_span:
                out.append(Run(int(codes[start]), start, i))
            start = i
    return out


def encode_aligned(vocab: BPEVocab, runs: list[Run]) -> list[tuple[int, int, int]]:
    """``[(token_id, start_transition, end_transition), ...]`` for one episode.

    ``BPEVocab.encode_span`` returns ids only, and it silently drops symbols missing
    from the lookup, so walking its output with token lengths would mis-align exactly
    when something is wrong. This repeats the greedy merge while carrying spans, so
    the alignment cannot drift from the ids.
    """
    symbols = [((r.symbol,), r.start, r.end) for r in runs]
    for left, right in vocab.merges:
        i = 0
        merged: list[tuple[tuple, int, int]] = []
        while i < len(symbols):
            if (
                i + 1 < len(symbols)
                and symbols[i][0] == left
                and symbols[i + 1][0] == right
            ):
                merged.append((left + right, symbols[i][1], symbols[i + 1][2]))
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        symbols = merged
    return [
        (vocab.id_of(sym), lo, hi) for sym, lo, hi in symbols if sym in vocab._lookup
    ]


def _score(true_b: set[int], pred_b: set[int], tolerance: int) -> BoundaryScore:
    """Greedy one-to-one match within ``tolerance``, as in repeatability.py.

    One-to-one matters: without it a burst of predicted boundaries around one true
    boundary would all count as true positives and precision would be inflated.
    """
    matched: set[int] = set()
    tp = fn = 0
    for t in sorted(true_b):
        hits = [p for p in pred_b if abs(p - t) <= tolerance and p not in matched]
        if hits:
            matched.add(min(hits, key=lambda p: abs(p - t)))
            tp += 1
        else:
            fn += 1
    return BoundaryScore(tp, len(pred_b) - len(matched), fn, tolerance)


def episode_report(
    codes: Sequence[int],
    ep: Episode,
    meta: TaskMeta,
    vocab: BPEVocab,
    min_span: int = 3,
    tolerance: int = 8,
) -> dict:
    segments = sg.segments_from_track(meta.episode_labels(ep.epis_idx), ep.exec_start)
    true_b = {s.start - ep.exec_start for s in segments if s.start > ep.exec_start}

    runs = runs_with_spans(codes, min_span)
    run_b = {r.start for r in runs if r.start > 0}
    tokens = encode_aligned(vocab, runs)
    tok_b = {lo for _, lo, _ in tokens if lo > 0}

    return {
        "true_events": len(segments),
        "run_symbols": len(runs),
        "bpe_tokens": len(tokens),
        "runs": _score(true_b, run_b, tolerance),
        "bpe": _score(true_b, tok_b, tolerance),
    }


def report(
    streams: dict[int, Sequence[int]],
    episodes: dict[int, Episode],
    meta: TaskMeta,
    vocab: BPEVocab,
    min_span: int = 3,
    tolerance: int = 8,
) -> dict:
    """Aggregate over episodes. Counts are pooled before the ratio, not averaged.

    Averaging per-episode precision would weight a 200-frame episode the same as an
    800-frame one and quietly change the number.
    """
    agg = {"runs": [0, 0, 0], "bpe": [0, 0, 0]}
    n_true = n_runs = n_tokens = 0
    for epis_idx, codes in streams.items():
        r = episode_report(codes, episodes[epis_idx], meta, vocab, min_span, tolerance)
        n_true += r["true_events"]
        n_runs += r["run_symbols"]
        n_tokens += r["bpe_tokens"]
        for key in ("runs", "bpe"):
            s = r[key]
            agg[key][0] += s.tp
            agg[key][1] += s.fp
            agg[key][2] += s.fn

    n = max(len(streams), 1)
    out = {
        "episodes": len(streams),
        "true_events_per_episode": n_true / n,
        "run_symbols_per_episode": n_runs / n,
        "bpe_tokens_per_episode": n_tokens / n,
        "vocab_size": vocab.size,
        "merges": len(vocab.merges),
        "tolerance": tolerance,
    }
    for key in ("runs", "bpe"):
        s = BoundaryScore(*agg[key], tolerance)
        out[f"{key}_precision"] = s.precision
        out[f"{key}_recall"] = s.recall
        out[f"{key}_f1"] = s.f1
    # The number the stage has to justify: does merging move F1 at all?
    out["bpe_f1_gain"] = out["bpe_f1"] - out["runs_f1"]
    out["over_segmentation_runs"] = out["run_symbols_per_episode"] / max(
        out["true_events_per_episode"], 1e-9
    )
    out["over_segmentation_bpe"] = out["bpe_tokens_per_episode"] / max(
        out["true_events_per_episode"], 1e-9
    )
    return out
