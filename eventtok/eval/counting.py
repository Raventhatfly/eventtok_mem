"""Counting, measured without circularity.

The rule that makes this valid: **the pattern is chosen on a train split, using
train N only, then its occurrence count is evaluated on held-out episodes whose N
is never consulted during selection.**

Three earlier attempts were all broken, in ways worth recording so they are not
repeated:

1. ``max(token_counts) == N`` — counted the *most frequent* token, never
   identifying which token means the repeated event, and counted **tokens** where
   one event can span several. Gave 48%, against a shuffled-label baseline of
   31-37%, i.e. indistinguishable from chance.
2. ``best_repeating_ngram(codes, target=N)`` — selected the n-gram *by* its
   closeness to N and then scored it against N. Circular; its "29/40" was
   meaningless.
3. Reporting either without a chance baseline at all. The baseline here is high
   (~37%) because the N distribution and the count distribution both pile up on
   2-3, so accidental agreement is common.

Done properly on SwingXtimes: a single run-symbol selected on 50 train episodes
reaches **49/50 = 98%** on the 50 held-out episodes, against 37% (sd 6) for
shuffled labels.

Note what that implies about the pipeline: counting works on **runs of the raw
code stream**, with no BPE at all. Whether the BPE stage adds anything to counting
is an open question, and it should not be assumed.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def runs_of(codes: Sequence[int], min_span: int = 3) -> list[int]:
    """Collapse a code stream to its run symbols, dropping runs shorter than
    ``min_span`` (jitter suppression, as in the streaming tokenizer)."""
    if not codes:
        return []
    out: list[int] = []
    cur = [codes[0]]
    for x in codes[1:]:
        if x == cur[-1]:
            cur.append(x)
        else:
            if len(cur) >= min_span:
                out.append(cur[0])
            cur = [x]
    if len(cur) >= min_span:
        out.append(cur[0])
    return out


def _grams(seq: Sequence[int], length: int) -> list[tuple[int, ...]]:
    return [tuple(seq[i : i + length]) for i in range(len(seq) - length + 1)]


@dataclass
class CountPattern:
    gram: tuple[int, ...]
    length: int
    train_accuracy: float
    train_correlation: float


def select_pattern(
    runs: dict[int, list[int]],
    counts: dict[int, int],
    train_ids: Sequence[int],
    max_len: int = 6,
) -> CountPattern | None:
    """Pick the pattern whose occurrence count best matches N **on train only**."""
    best: CountPattern | None = None
    for length in range(1, max_len + 1):
        candidates = {g for e in train_ids for g in set(_grams(runs[e], length))}
        for gram in candidates:
            occ = np.array(
                [Counter(_grams(runs[e], length))[gram] for e in train_ids], dtype=float
            )
            n = np.array([counts[e] for e in train_ids], dtype=float)
            if occ.std() < 1e-9:
                continue
            acc = float((occ == n).mean())
            corr = float(np.corrcoef(occ, n)[0, 1])
            if best is None or acc > best.train_accuracy:
                best = CountPattern(gram, length, acc, corr)
    return best


def evaluate(
    runs: dict[int, list[int]],
    counts: dict[int, int],
    pattern: CountPattern,
    test_ids: Sequence[int],
) -> dict:
    """Count the *fixed* pattern on held-out episodes and compare to N."""
    predicted = {e: Counter(_grams(runs[e], pattern.length))[pattern.gram] for e in test_ids}
    exact = sum(1 for e in test_ids if predicted[e] == counts[e])
    over = sum(1 for e in test_ids if predicted[e] > counts[e])
    under = sum(1 for e in test_ids if predicted[e] < counts[e])
    mae = float(np.mean([abs(predicted[e] - counts[e]) for e in test_ids]))
    return {
        "n_test": len(test_ids),
        "exact": exact,
        "over": over,
        "under": under,
        "accuracy": exact / max(len(test_ids), 1),
        "mae": mae,
        "predicted": predicted,
    }


def shuffled_baseline(
    runs: dict[int, list[int]],
    counts: dict[int, int],
    pattern: CountPattern,
    test_ids: Sequence[int],
    trials: int = 400,
    seed: int = 0,
) -> tuple[float, float]:
    """Chance level: same predictions, N labels permuted. Returns (mean, sd).

    Not optional. The marginals overlap heavily, so accidental agreement runs
    around 37% on SwingXtimes — an accuracy reported without this is unreadable.
    """
    rng = random.Random(seed)
    predicted = [Counter(_grams(runs[e], pattern.length))[pattern.gram] for e in test_ids]
    labels = [counts[e] for e in test_ids]
    scores = []
    for _ in range(trials):
        shuffled = labels[:]
        rng.shuffle(shuffled)
        scores.append(
            sum(1 for p, n in zip(predicted, shuffled) if p == n) / max(len(test_ids), 1)
        )
    return float(np.mean(scores)), float(np.std(scores))


def split(episode_ids: Sequence[int], frac: float = 0.5, seed: int = 0):
    ids = list(episode_ids)
    random.Random(seed).shuffle(ids)
    cut = int(len(ids) * frac)
    return ids[:cut], ids[cut:]
