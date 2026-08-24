"""The token source must be explicit and must actually change the codes.

Guards the failure that recurred twice: experiments silently building action-only
streams while claiming to measure a method that uses action and vision.
"""

from __future__ import annotations

import numpy as np
import pytest

from eventtok import paths
from eventtok.data import repack
from eventtok.data.index import RoboMMEIndex
from eventtok.data.meta import TaskMeta
from eventtok.models.streams import build_streams

TASK = "SwingXtimes"


@pytest.fixture(scope="module")
def setup():
    paths.check_root()
    index = RoboMMEIndex()
    eps = index.by_task(TASK)[:16]
    return TaskMeta(TASK), eps, repack.EpisodeFeatures(TASK, "2x2")


def test_token_source_is_required(setup) -> None:
    meta, eps, feats = setup
    with pytest.raises(TypeError):
        build_streams(meta, eps[:8], eps)  # no `tokens` -> must not default


def test_unknown_source_is_rejected(setup) -> None:
    meta, eps, feats = setup
    with pytest.raises(ValueError, match="action"):
        build_streams(meta, eps[:8], eps, tokens="vision-only", features=feats)


def test_multimodal_needs_features(setup) -> None:
    meta, eps, _ = setup
    with pytest.raises(ValueError, match="features"):
        build_streams(meta, eps[:8], eps, tokens="action+vision", features=None)


def test_the_two_sources_give_different_codes(setup) -> None:
    """If these agreed, the multimodal path would be a silent no-op."""
    meta, eps, feats = setup
    a = build_streams(meta, eps[:8], eps, tokens="action", features=feats)
    m = build_streams(meta, eps[:8], eps, tokens="action+vision", features=feats)
    assert set(a) == set(m)
    agree = np.mean([x == y for e in a for x, y in zip(a[e], m[e])])
    assert agree < 0.5, f"streams agree {agree:.1%}; vision is not affecting the codes"


def test_streams_cover_every_transition(setup) -> None:
    meta, eps, feats = setup
    for src in ("action", "action+vision"):
        s = build_streams(meta, eps[:8], eps, tokens=src, features=feats)
        for e in eps:
            lo, hi = meta.rows(e.epis_idx)
            assert len(s[e.epis_idx]) == hi - lo, f"{src} {e.epis_idx}"
