"""The one way to build a code stream. Import this, never a tokenizer directly.

This module exists because of a repeated failure, not for tidiness. Every experiment in
this project -- counting, extrapolation, boundaries, the consumption probe, the
diffusion policy -- built its code stream by calling ``KMeansTokenizer``, which is
action-only and is documented in its own docstring as the OAT-shaped *control*. So
every result was an OAT result, and the method this project is about was never
measured. The user caught it twice; the second time, six call sites were still wrong.

The cause was interface convenience: ``KMeansTokenizer.stream_for_episode()`` existed
and the multimodal equivalent did not, so the function with the right signature won over
the function with the right semantics. :func:`build_streams` removes that asymmetry --
both paths are one call with the same return type, and ``tokens`` is a required
argument, so a caller has to state what it is tokenizing rather than inherit a default.
"""

from __future__ import annotations

from typing import Literal

from ..data.meta import TaskMeta

TokenSource = Literal["action", "action+vision"]


def build_streams(
    meta: TaskMeta,
    fit_episodes,
    all_episodes,
    tokens: TokenSource,
    k: int = 16,
    seed: int = 0,
    features=None,
    horizon: int = 20,
    vision_weight: float = 1.0,
) -> dict[int, list[int]]:
    """``{epis_idx: [code, ...]}``, fitted on ``fit_episodes`` only.

    Args:
        tokens: ``"action"`` is the OAT control; ``"action+vision"`` is the method.
            Required on purpose -- there is no default to drift back to.
        features: per-episode visual features, required for ``"action+vision"``.
    """
    if tokens == "action":
        from .kmeans import KMeansTokenizer

        km = KMeansTokenizer(k, seed=seed).fit(meta, fit_episodes)
        return {e.epis_idx: km.stream_for_episode(meta, e) for e in all_episodes}

    if tokens == "action+vision":
        if features is None:
            raise ValueError(
                "action+vision tokens need `features`; pass "
                "repack.EpisodeFeatures(task, scale) or a DINO equivalent"
            )
        from .multimodal import MultimodalTokenizer

        mm = MultimodalTokenizer(
            n_clusters=k, horizon=horizon, vision_weight=vision_weight, seed=seed
        ).fit(meta, fit_episodes, features)
        return mm.stream_for_episodes(meta, all_episodes, features)

    raise ValueError(f"tokens must be 'action' or 'action+vision', got {tokens!r}")
