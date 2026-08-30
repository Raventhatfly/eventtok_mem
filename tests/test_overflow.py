"""Evicting a token must not destroy its count.

The project's claim is count preservation. Plain FIFO on a full log deletes exactly
that: swing ten times with a five-token budget and the count reads five. Measured on
a 1300-step failing rollout, an oscillating policy produces 93-156 tokens against a
64 budget, so this path is reached in practice -- and it used to drop history with no
error and no flag.
"""

from __future__ import annotations

import numpy as np

from eventtok.bpe import build_vocab as bpe
from eventtok.rollout.online_log import OnlineEventLog


def _log(max_log: int, k: int = 8):
    """A log whose codes come from a trivial 1-D centroid set, so codes are predictable."""
    centroids = np.arange(k, dtype=np.float32).reshape(k, 1) * 10.0
    vocab = bpe.train([[0, 1, 2, 3]], vocab_size=16, min_frequency=99, max_token_length=4)
    scale = np.ones(1, dtype=np.float32)
    return OnlineEventLog(centroids, scale, vocab, chunk=1, min_span=1, max_log=max_log)


def _drive(log, seq, reps):
    """Push each value in `seq` `reps` times so every value forms its own run."""
    for _ in range(reps):
        for val in seq:
            log.push(np.array([val * 10.0], dtype=np.float32), np.zeros(1, dtype=np.float32))


def test_total_counts_survive_eviction() -> None:
    """The whole point: a small window must not change the total."""
    big, small = _log(max_log=10_000), _log(max_log=3)
    for lg in (big, small):
        _drive(lg, [0, 1], reps=12)
    assert small.overflowed() > 0, "the window must actually have overflowed"
    assert np.array_equal(small.total_counts(), big.total_counts()), (
        "totals must not depend on the budget"
    )


def test_visible_window_is_bounded_and_recent() -> None:
    small = _log(max_log=3)
    _drive(small, [0, 1], reps=12)
    assert len(small.tokens()) <= 3
    # what remains is the tail, not the head
    big = _log(max_log=10_000)
    _drive(big, [0, 1], reps=12)
    assert small.tokens() == big.tokens()[-3:]


def test_counts_under_reports_but_total_does_not() -> None:
    """`counts` is window-only by design; `total_counts` is the honest one."""
    small = _log(max_log=3)
    _drive(small, [0, 1], reps=12)
    assert small.counts().sum() <= 3
    assert small.total_counts().sum() == small.counts().sum() + small.overflow().sum()


def test_no_overflow_when_it_fits() -> None:
    lg = _log(max_log=10_000)
    _drive(lg, [0, 1], reps=3)
    assert lg.overflowed() == 0
    assert lg.overflow().sum() == 0
    assert np.array_equal(lg.total_counts(), lg.counts())
