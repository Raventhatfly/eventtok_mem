"""The online log must equal the offline log. Otherwise the rollout measures something
other than the system every offline number describes.

Two properties, both of which could break silently:

* the final token sequence matches what the offline pipeline produces for the same
  episode, and
* the prefix at time t contains exactly the spans that closed by t -- no future leak,
  which is the failure that would inflate the counting results.
"""

from __future__ import annotations

import numpy as np
import pytest

from eventtok import paths
from eventtok.bpe import build_vocab as bpe
from eventtok.data.index import RoboMMEIndex
from eventtok.data.meta import TaskMeta
from eventtok.eval.bpe_boundaries import encode_aligned, runs_with_spans
from eventtok.models.kmeans import KMeansTokenizer
from eventtok.rollout.online_log import OnlineEventLog, stable_prefix_encode

TASK = "SwingXtimes"
CHUNK = 20
MIN_SPAN = 3


@pytest.fixture(scope="module")
def fixture():
    paths.check_root()
    index = RoboMMEIndex()
    eps = index.by_task(TASK)[:12]
    meta = TaskMeta(TASK)
    km = KMeansTokenizer(16, seed=0).fit(meta, eps[:6])
    streams = {e.epis_idx: km.stream_for_episode(meta, e) for e in eps}
    corpus = [[r.symbol for r in runs_with_spans(streams[e.epis_idx], MIN_SPAN)]
              for e in eps[:6]]
    vocab = bpe.train(corpus, vocab_size=128, min_frequency=3, max_token_length=20)
    return index, meta, km, vocab, streams, eps


def _online_for_episode(meta, km, vocab, ep, upto=None):
    """Replay an episode's recorded actions through the online log."""
    lo, hi = meta.rows(ep.epis_idx)
    n = hi - lo if upto is None else min(upto, hi - lo)
    log = OnlineEventLog(km.centroids, meta.action_scale, vocab,
                         chunk=CHUNK, min_span=MIN_SPAN, max_log=10**6)
    for t in range(n):
        # The action actually executed at t is the first step of the chunk at t.
        log.push(meta.actions[lo + t][0], meta.state[lo + t])
    return log


def test_online_codes_match_offline(fixture) -> None:
    """Per-chunk k-means codes must be identical to the offline stream."""
    _, meta, km, vocab, streams, eps = fixture
    ep = eps[0]
    lo, hi = meta.rows(ep.epis_idx)
    log = _online_for_episode(meta, km, vocab, ep)
    offline = streams[ep.epis_idx]
    # Online can only code a chunk once its CHUNK actions have been executed, so the
    # chunks it can code start at t = 0 .. n - CHUNK: one fewer than the offline
    # stream's per-row code for every row, plus the one at t = n - CHUNK itself.
    assert len(log._codes) == max((hi - lo) - CHUNK + 1, 0)
    assert log._codes == list(offline[: len(log._codes)])


def test_online_tokens_match_offline(fixture) -> None:
    """The BPE token sequence must match, not just the codes."""
    _, meta, km, vocab, streams, eps = fixture
    for ep in eps[:4]:
        log = _online_for_episode(meta, km, vocab, ep)
        truncated = list(streams[ep.epis_idx])[: len(log._codes)]
        runs = [r.symbol for r in runs_with_spans(truncated, MIN_SPAN)]
        # The stabilised encoding, which is what the offline cache stores. Comparing
        # against encode_aligned would only assert that both paths share the same
        # future leak.
        offline = stable_prefix_encode(vocab, runs)
        assert log.tokens() == offline, f"episode {ep.epis_idx}"


def test_prefix_never_leaks_the_future(fixture) -> None:
    """The log at time t must be a prefix of the log at any later time.

    A violation means a token appeared and then changed, which is what an incremental
    BPE encoder would do when a merge spans the boundary -- and it would mean the
    policy saw information about actions it had not taken yet.
    """
    _, meta, km, vocab, streams, eps = fixture
    ep = eps[0]
    lo, hi = meta.rows(ep.epis_idx)
    n = hi - lo
    prev: list[int] = []
    for t in range(CHUNK, n, 25):
        cur = _online_for_episode(meta, km, vocab, ep, upto=t).tokens()
        assert cur[: len(prev)] == prev or len(cur) < len(prev) is False, (
            f"log at t={t} is not an extension of the earlier log"
        )
        prev = cur


def test_empty_log_at_episode_start(fixture) -> None:
    """Before any chunk closes there is nothing to condition on, and that is a state,
    not an error."""
    _, meta, km, vocab, _, eps = fixture
    log = _online_for_episode(meta, km, vocab, eps[0], upto=CHUNK - 1)
    assert log.tokens() == []
    assert len(log) == 0
    assert log.counts(vocab.size).sum() == 0
