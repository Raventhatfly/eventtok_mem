"""The rollout log must equal the cached log the policy trained on.

This is the test that decides whether a success rate means anything. The cache is built
from recorded chunks with whole-episode information available; the rollout rebuilds the
same quantity from executed actions arriving one at a time. If the two disagree, the
policy is being tested on a signal it never saw, and it fails silently -- a log built by
the wrong rules still looks like a log.
"""

from __future__ import annotations

import numpy as np
import pytest

from eventtok.data import repack
from eventtok.data.index import RoboMMEIndex
from eventtok.data.meta import TaskMeta
from eventtok.pi05.online import EventTokenizerBundle, RolloutEventLog, tokenizer_path
from eventtok.pi05.tokens import EventLogCache, cache_path

TAG = "joint_k64"
TASK = "SwingXtimes"
pytestmark = pytest.mark.skipif(
    not (tokenizer_path(TAG).is_file() and cache_path(TASK, TAG).is_file()),
    reason=f"no {TAG} tokenizer/cache",
)


@pytest.fixture(scope="module")
def parts():
    return (
        EventTokenizerBundle(TAG),
        EventLogCache(TASK, TAG),
        TaskMeta(TASK),
        RoboMMEIndex(),
        repack.EpisodeFeatures(TASK, "2x2"),
    )


def _replay(bundle, meta, feats, ep, n_steps=None):
    """Drive the rollout log with an episode's recorded behaviour, one step at a time."""
    lo, hi = meta.rows(ep.epis_idx)
    n = hi - lo if n_steps is None else min(n_steps, hi - lo)
    log = RolloutEventLog(bundle, TASK)
    arr = feats[int(ep.epis_idx)]
    off = ep.exec_start if getattr(feats, "indexes_absolute_frames", True) else 0
    out = []
    for t in range(n):
        # The action executed at t is the first step of the chunk recorded at t; the
        # frame is the visual feature at the same execution step.
        log.push(meta.actions[lo + t][0], meta.state[lo + t],
                 np.asarray(arr[min(t + off, len(arr) - 1)], dtype=np.float32).ravel())
        out.append(log.tokens())
    return out


def test_rollout_tokens_match_the_cache(parts):
    """Exactly, at every frame -- not approximately, and not with a lag.

    The cache used to reveal a run at the frame it ended, but row t's code is computed
    from the visual feature at t+chunk, so that token could not have existed yet. With
    the reveal shifted to end+chunk the two agree at every frame; before the shift, every
    frame in the 20 steps after each run boundary disagreed.
    """
    bundle, cache, meta, index, feats = parts
    eps = [e for e in index.by_task(TASK) if cache.is_train_episode(e.epis_idx)][:3]
    assert eps, "no train episodes"
    for ep in eps:
        got = _replay(bundle, meta, feats, ep)
        lo, hi = meta.rows(ep.epis_idx)
        mismatch = [
            t for t in range(hi - lo)
            if got[t] != list(cache[(ep.epis_idx, t)][0][: cache[(ep.epis_idx, t)][1]])
        ]
        assert not mismatch, (
            f"episode {ep.epis_idx}: {len(mismatch)}/{hi - lo} frames disagree, "
            f"first at t={mismatch[0]}: rollout {got[mismatch[0]]} vs cache "
            f"{list(cache[(ep.epis_idx, mismatch[0])][0][:cache[(ep.epis_idx, mismatch[0])][1]])}"
        )


def test_log_is_append_only(parts):
    bundle, cache, meta, index, feats = parts
    ep = next(e for e in index.by_task(TASK) if cache.is_train_episode(e.epis_idx))
    seq = _replay(bundle, meta, feats, ep)
    for a, b in zip(seq, seq[1:]):
        assert b[: len(a)] == a, "a token was revised after being shown to the policy"


def test_nothing_before_the_first_chunk_closes(parts):
    bundle, cache, meta, index, feats = parts
    ep = next(e for e in index.by_task(TASK) if cache.is_train_episode(e.epis_idx))
    seq = _replay(bundle, meta, feats, ep, n_steps=bundle.chunk + 1)
    assert all(not s for s in seq[: bundle.chunk]), (
        "the log is non-empty before any chunk could have executed"
    )
