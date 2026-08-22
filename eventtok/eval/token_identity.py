"""Can a BPE **event token** be named? The number the memory claim actually rests on.

Every identity number in this project so far was computed on per-frame k-means cluster
ids. That answers "given this frame's code, which event is this frame in". It does not
answer "given this event token, which event is it" -- and the event token is the thing
that would go into the log, get rendered to text, and be read by a policy. The two can
differ, for two reasons that both matter here:

* **Straddling.** A BPE token spans several run symbols and therefore several frames.
  Boundary placement is poor (F1 <= 0.46, precision <= 0.35), so a token can begin
  inside one event and end inside the next. Such a token has no single correct label,
  and the per-frame metric never has to confront that.
* **Weighting.** Per-frame accuracy is length-weighted, so a 200-frame event counts ten
  times a 20-frame one. Per-token weights each occurrence once, which is what a log of
  events implies.

Definitions used here, both stated because both are choices:

  a token instance's true label   the majority observable label over the frames it
                                  spans. Generous -- it gives a straddling token the
                                  benefit of its larger half.
  purity                          the fraction of the token's frames carrying that
                                  majority label. 1.0 means the token sits inside one
                                  event; lower means it straddles. Reported alongside
                                  the accuracy, because a high accuracy over impure
                                  tokens is measuring a coin flip that landed well.

The naming table is fitted on train episodes (token id -> majority label over its
instances) and scored on held-out ones, one vote per token instance, against the
majority-label rate among held-out instances.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

from ..bpe.build_vocab import BPEVocab
from ..data import subgoals as sg
from ..data.index import Episode
from ..data.meta import TaskMeta
from .bpe_boundaries import encode_aligned, runs_with_spans


def frame_labels(ep: Episode, meta: TaskMeta) -> list[str]:
    """Observable label per execution frame, aligned to the code stream."""
    segments = sg.segments_from_track(meta.episode_labels(ep.epis_idx), ep.exec_start)
    out: list[str] = []
    for s in segments:
        out.extend([sg.observable_label(s.label)] * (s.end - s.start))
    return out


def token_instances(
    codes: Sequence[int],
    ep: Episode,
    meta: TaskMeta,
    vocab: BPEVocab,
    min_span: int = 3,
) -> list[tuple[int, str, float]]:
    """``[(token_id, majority_label, purity), ...]`` for one episode."""
    labels = frame_labels(ep, meta)
    runs = runs_with_spans(codes, min_span)
    out = []
    for token, lo, hi in encode_aligned(vocab, runs):
        window = labels[lo:hi]
        if not window:
            continue
        label, n = Counter(window).most_common(1)[0]
        out.append((token, label, n / len(window)))
    return out


def report(
    streams: dict[int, Sequence[int]],
    episodes: dict[int, Episode],
    meta: TaskMeta,
    vocab: BPEVocab,
    train_ids: set[int],
    min_span: int = 3,
) -> dict:
    per_ep = {
        idx: token_instances(codes, episodes[idx], meta, vocab, min_span)
        for idx, codes in streams.items()
    }

    fit: dict[int, Counter] = {}
    for idx, insts in per_ep.items():
        if idx in train_ids:
            for token, label, _ in insts:
                fit.setdefault(token, Counter())[label] += 1
    naming = {t: c.most_common(1)[0][0] for t, c in fit.items()}

    test = [(t, l, p) for idx, insts in per_ep.items() if idx not in train_ids
            for t, l, p in insts]
    if not test:
        return {"token_instances": 0}

    truth = [l for _, l, _ in test]
    fallback = Counter(truth).most_common(1)[0][0]
    correct = sum(1 for t, l, _ in test if naming.get(t, fallback) == l)
    majority = Counter(truth).most_common(1)[0][1] / len(test)
    purity = np.array([p for _, _, p in test])

    return {
        "token_instances": len(test),
        "token_accuracy": correct / len(test),
        "token_majority": majority,
        "token_gain": correct / len(test) - majority,
        "mean_purity": float(purity.mean()),
        "pure_fraction": float((purity >= 0.999).mean()),
        "straddle_fraction": float((purity < 0.999).mean()),
        "named_tokens": len(naming),
        "unseen_token_rate": float(
            sum(1 for t, _, _ in test if t not in naming) / len(test)
        ),
    }
