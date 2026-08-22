"""Pin the span alignment. A silent drift here would corrupt every boundary number.

``encode_aligned`` repeats BPE's greedy merge while carrying frame spans. If it
diverged from ``BPEVocab.encode_span`` the token ids and the spans would describe
different segmentations, and the boundary metric would be measuring nothing --
without failing.
"""

from __future__ import annotations

import pytest

from eventtok.bpe import build_vocab as bpe
from eventtok.eval.bpe_boundaries import Run, encode_aligned, runs_with_spans, _score


def test_runs_keep_exact_spans() -> None:
    codes = [1, 1, 1, 2, 2, 2, 2, 1, 1, 1]
    runs = runs_with_spans(codes, min_span=3)
    assert [(r.symbol, r.start, r.end) for r in runs] == [
        (1, 0, 3),
        (2, 3, 7),
        (1, 7, 10),
    ]
    # Spans tile the stream with no gap or overlap.
    for a, b in zip(runs, runs[1:]):
        assert a.end == b.start


def test_short_runs_are_dropped_and_leave_gaps() -> None:
    """Jitter suppression drops runs, so spans stop tiling -- by design.

    Boundaries come from run starts, so a dropped run must not shift the surviving
    spans; it just removes a candidate boundary.
    """
    codes = [1, 1, 1, 9, 2, 2, 2]        # the lone 9 is below min_span
    runs = runs_with_spans(codes, min_span=3)
    assert [(r.symbol, r.start, r.end) for r in runs] == [(1, 0, 3), (2, 4, 7)]


def test_runs_of_agrees_with_runs_with_spans() -> None:
    """The two run extractors must not drift apart."""
    from eventtok.eval.counting import runs_of

    codes = [3, 3, 3, 3, 1, 1, 5, 5, 5, 2, 2, 2, 2, 2]
    assert [r.symbol for r in runs_with_spans(codes, 3)] == runs_of(codes, 3)


def test_encode_aligned_matches_encode_span_ids() -> None:
    """Same ids as the vocab's own encoder, plus spans that cover the input."""
    corpus = [[1, 2, 1, 2, 3], [1, 2, 1, 2, 3], [1, 2, 3, 1, 2]]
    vocab = bpe.train(corpus, vocab_size=32, min_frequency=2, max_token_length=8)

    seq = [1, 2, 1, 2, 3]
    runs = [Run(c, i, i + 1) for i, c in enumerate(seq)]
    aligned = encode_aligned(vocab, runs)

    assert [t for t, _, _ in aligned] == vocab.encode_span(seq)
    # Spans are contiguous and cover the whole input exactly once.
    assert aligned[0][1] == 0
    assert aligned[-1][2] == len(seq)
    for (_, _, end), (_, nxt, _) in zip(aligned, aligned[1:]):
        assert end == nxt
    # Each token's span width equals the number of base symbols it merged.
    for token, lo, hi in aligned:
        assert hi - lo == len(vocab.decode_token(token))


def test_encode_aligned_preserves_real_frame_spans() -> None:
    """Spans of unequal width -- the real case -- compose correctly."""
    corpus = [[4, 7, 4, 7], [4, 7, 4, 7], [4, 7, 9]]
    vocab = bpe.train(corpus, vocab_size=32, min_frequency=2, max_token_length=8)
    runs = [Run(4, 0, 10), Run(7, 10, 35), Run(4, 35, 40), Run(7, 40, 100)]
    aligned = encode_aligned(vocab, runs)
    assert aligned[0][1] == 0 and aligned[-1][2] == 100
    for (_, _, end), (_, nxt, _) in zip(aligned, aligned[1:]):
        assert end == nxt


def test_score_matching_is_one_to_one() -> None:
    """A burst of predictions near one true boundary must not all count as hits.

    Without one-to-one matching, precision inflates and an over-segmenting stream
    scores like a good one -- which is the failure this whole metric exists to catch.
    """
    s = _score(true_b={50}, pred_b={48, 49, 50, 51, 52}, tolerance=8)
    assert (s.tp, s.fp, s.fn) == (1, 4, 0)
    assert s.precision == pytest.approx(0.2)
    assert s.recall == pytest.approx(1.0)


def test_score_counts_misses() -> None:
    s = _score(true_b={10, 100}, pred_b={11}, tolerance=8)
    assert (s.tp, s.fp, s.fn) == (1, 0, 1)
