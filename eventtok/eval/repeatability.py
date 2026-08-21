"""Does the code stream expose repetition? The measurement that decides the design.

Two properties, reported separately because they can come apart:

**Stability** — how often the code changes *within* one annotated event. A discrete
event vocabulary sits near 0%. A fine-grained trajectory code sits high, and then
"the code changed" is useless as a boundary signal.

**Recoverability** — whether a recurring n-gram occurs exactly N times, where N is
the episode's target repetition count. This can hold even when stability fails,
because a repeated *motion* leaves a repeated *subsequence*. That is the property
BPE actually needs.

The first measured run (|C| = 512, 8 epochs) gave 27.5% within-event change and
8/12 exact n-gram matches, with all four misses overcounting by exactly one. The
diagnosis was that 512 codes over a smooth trajectory is close to a unique code
per timestep, which makes exact repeats fragile — hence the codebook sweep. PRISE
used an alphabet of 10, Genie 8, IGOR 32.

Deliberately *not* included: "some code occurs exactly N times". With ~120 live
codes and N in {1,2,3} that is true by chance almost always, and it looks like
evidence when it is not.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from ..data import subgoals as sg
from ..data.index import Episode
from ..data.meta import TaskMeta


@dataclass
class Stability:
    changes: int
    transitions: int

    @property
    def change_rate(self) -> float:
        return self.changes / max(self.transitions, 1)

    @property
    def mean_run(self) -> float:
        return self.transitions / max(self.changes, 1)


def within_event_stability(
    codes: Sequence[int], ep: Episode, meta: TaskMeta
) -> Stability:
    """How often the code changes inside one annotated event."""
    segments = sg.segments_from_track(meta.episode_labels(ep.epis_idx), ep.exec_start)
    changes = total = 0
    for s in segments:
        lo = max(s.start - ep.exec_start, 0)
        hi = min(s.end - ep.exec_start, len(codes))
        span = codes[lo:hi]
        if len(span) < 2:
            continue
        total += len(span) - 1
        changes += sum(1 for a, b in zip(span, span[1:]) if a != b)
    return Stability(changes, total)


def best_repeating_ngram(
    codes: Sequence[int], target: int, min_len: int = 4, max_len: int = 16
) -> tuple[tuple[int, ...], int, int] | None:
    """The recurring n-gram whose occurrence count is closest to ``target``.

    Returns ``(gram, occurrences, length)``. Ties prefer the longer gram, since a
    longer match is less likely to be coincidental.
    """
    best = None
    for length in range(min_len, max_len + 1):
        if length > len(codes):
            break
        grams = Counter(
            tuple(codes[i : i + length]) for i in range(len(codes) - length + 1)
        )
        for gram, count in grams.most_common(8):
            if count < 2:
                continue
            if best is None:
                best = (gram, count, length)
                continue
            better = abs(count - target) < abs(best[1] - target)
            same_and_longer = (
                abs(count - target) == abs(best[1] - target) and length > best[2]
            )
            if better or same_and_longer:
                best = (gram, count, length)
    return best


def report(
    streams: dict[int, Sequence[int]],
    episodes: dict[int, Episode],
    meta: TaskMeta,
    min_len: int = 4,
    max_len: int = 16,
) -> dict:
    """Aggregate both properties over a set of episodes."""
    changes = total = 0
    exact = over = under = 0
    rows = []
    for epis_idx, codes in streams.items():
        ep = episodes[epis_idx]
        stab = within_event_stability(codes, ep, meta)
        changes += stab.changes
        total += stab.transitions

        found = best_repeating_ngram(codes, ep.count or 0, min_len, max_len)
        occ = found[1] if found else 0
        if ep.count is not None:
            if occ == ep.count:
                exact += 1
            elif occ > ep.count:
                over += 1
            else:
                under += 1
        rows.append(
            {
                "episode": epis_idx,
                "N": ep.count,
                "len": len(codes),
                "gram_len": found[2] if found else None,
                "occurrences": occ,
                "change_rate": stab.change_rate,
            }
        )

    n = max(len(streams), 1)
    return {
        "within_event_change_rate": changes / max(total, 1),
        "mean_code_run": total / max(changes, 1),
        "ngram_exact": exact,
        "ngram_over": over,
        "ngram_under": under,
        "episodes": n,
        "exact_frac": exact / n,
        "rows": rows,
    }
