"""Does the code stream expose repetition? The measurement that decides the design.

Two properties, reported separately because they can come apart:

**Stability** — how often the code changes *within* one annotated event. A discrete
event vocabulary sits near 0%. A fine-grained trajectory code sits high, and then
"the code changed" is useless as a boundary signal.

**Always report a baseline on the same line.** Four numbers in this project were
misread for want of one: max-token counting (48% vs 31-37% chance, i.e. nothing),
a circular n-gram metric (chose the gram by the answer), counting with no chance
baseline at all, and MI/H read as an accuracy (53.9% MI is 80.5% accuracy against
a 49.6% majority). Use :func:`label_accuracy`, which returns its own baseline.

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
    """DO NOT USE FOR EVALUATION -- this is circular.

    It selects the n-gram whose occurrence count is closest to ``target`` (= N),
    then callers report how often that count equals N. Using the answer to pick
    the thing being tested guarantees a high score. The "29/40 n-grams match N"
    figure it produced was meaningless.

    Kept only for exploratory inspection of what patterns exist in a stream.
    For a real measurement use ``eventtok.eval.counting``: pick the pattern on a
    train split and evaluate its count on held-out episodes. Done that way,
    SwingXtimes gives 49/50 = 98% against a 37% (sd 6) shuffled-label baseline.
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

        # Circular by construction -- see best_repeating_ngram's docstring. Kept
        # in the dict for inspection only; do not report it as accuracy.
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
        # NOT an accuracy: the gram was chosen using N. Inspection only.
        "ngram_exact_CIRCULAR": exact,
        "ngram_over_CIRCULAR": over,
        "ngram_under_CIRCULAR": under,
        "episodes": n,
        "exact_frac": exact / n,
        "rows": rows,
    }


@dataclass
class BoundaryScore:
    tp: int
    fp: int
    fn: int
    tolerance: int

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-9)


def boundary_score(
    codes: Sequence[int], ep: Episode, meta: TaskMeta, tolerance: int = 8
) -> BoundaryScore:
    """Do code changes coincide with annotated event boundaries?

    **This is the metric that says whether anything is being segmented**, and it is
    not implied by within-event stability: a code constant over the whole episode
    scores 0% change rate — perfect by that measure — while segmenting nothing.

    Measured on action clusters over 40 SwingXtimes episodes, recall is 0.97-1.00
    but precision is 0.13-0.24. So the stream is a high-recall *candidate* set that
    over-segments 4-8x, not a segmentation. Merging candidates into events is the
    BPE stage's job; a code change is not an event boundary.
    """
    segments = sg.segments_from_track(meta.episode_labels(ep.epis_idx), ep.exec_start)
    true_b = {s.start - ep.exec_start for s in segments if s.start > ep.exec_start}
    pred_b = {i for i in range(1, len(codes)) if codes[i] != codes[i - 1]}

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


def label_mutual_information(
    codes: Sequence[int], ep: Episode, meta: TaskMeta, observable: bool = True
) -> tuple[list[int], list[str]]:
    """Paired (code, event label) samples, for MI over a whole task.

    Two canonicalisations, both necessary, both of which otherwise make a working
    tokenizer look broken:

    * repetition ordinals are stripped, or "for the second time" reads as a
      different event from "for the first time";
    * with ``observable=True`` (default) object identity is collapsed too, because
      on occlusion tasks the named object is hidden at the moment of the action and
      varies only across episodes. See :func:`subgoals.observable_label`.
    """
    segments = sg.segments_from_track(meta.episode_labels(ep.epis_idx), ep.exec_start)
    out_codes: list[int] = []
    out_labels: list[str] = []
    for s in segments:
        lo = max(s.start - ep.exec_start, 0)
        hi = min(s.end - ep.exec_start, len(codes))
        span = codes[lo:hi]
        out_codes.extend(int(c) for c in span)
        label = sg.observable_label(s.label) if observable else sg.canonical_label(s.label)
        out_labels.extend([label] * len(span))
    return out_codes, out_labels


def _entropy(values: Sequence) -> float:
    import math

    counts = Counter(values)
    n = sum(counts.values()) or 1
    return -sum((c / n) * math.log(c / n) for c in counts.values() if c)


def label_accuracy(
    codes: Sequence[int], labels: Sequence[str], train_mask: Sequence[bool]
) -> tuple[float, float]:
    """(accuracy, majority baseline) for predicting the event label from the code.

    **Prefer this over MI/H when reporting.** MI as a fraction of label entropy is
    hard to saturate, so it reads pessimistically: 53.9% MI on ButtonUnmask is
    80.5% label accuracy against a 49.6% majority baseline. Quoting the MI led to
    calling a working result "stuck".

    Each code is mapped to its majority label on the train frames, then evaluated
    on the rest. Always report the baseline alongside — the majority class is 50%
    on ButtonUnmask and 30% on SwingXtimes, so accuracy alone is unreadable.
    """
    codes = list(codes)
    labels = list(labels)
    train_mask = list(train_mask)
    per_code: dict[int, Counter] = {}
    for c, l, m in zip(codes, labels, train_mask):
        if m:
            per_code.setdefault(c, Counter())[l] += 1
    test_labels = [l for l, m in zip(labels, train_mask) if not m]
    if not test_labels:
        return 0.0, 0.0
    fallback = Counter(test_labels).most_common(1)[0][0]
    mapping = {c: cnt.most_common(1)[0][0] for c, cnt in per_code.items()}
    correct = sum(
        1
        for c, l, m in zip(codes, labels, train_mask)
        if not m and mapping.get(c, fallback) == l
    )
    majority = Counter(test_labels).most_common(1)[0][1] / len(test_labels)
    return correct / len(test_labels), majority


def mutual_information(a: Sequence, b: Sequence) -> float:
    import math

    n = len(a) or 1
    joint = Counter(zip(a, b))
    ca, cb = Counter(a), Counter(b)
    total = 0.0
    for (va, vb), c in joint.items():
        p = c / n
        total += p * math.log(p / ((ca[va] / n) * (cb[vb] / n)))
    return total


def full_report(
    streams: dict[int, Sequence[int]],
    episodes: dict[int, Episode],
    meta: TaskMeta,
    tolerance: int = 8,
) -> dict:
    """Everything at once: stability, boundary alignment, label MI, n-gram count."""
    base = report(streams, episodes, meta)

    tp = fp = fn = 0
    all_codes: list[int] = []
    all_labels: list[str] = []
    for epis_idx, codes in streams.items():
        ep = episodes[epis_idx]
        bs = boundary_score(codes, ep, meta, tolerance)
        tp, fp, fn = tp + bs.tp, fp + bs.fp, fn + bs.fn
        c, l = label_mutual_information(codes, ep, meta)
        all_codes.extend(c)
        all_labels.extend(l)

    overall = BoundaryScore(tp, fp, fn, tolerance)
    mi = mutual_information(all_codes, all_labels)
    h_label = _entropy(all_labels)
    base.update(
        {
            "boundary_precision": overall.precision,
            "boundary_recall": overall.recall,
            "boundary_f1": overall.f1,
            "boundary_tp": tp,
            "boundary_fp": fp,
            "boundary_fn": fn,
            "label_mi": mi,
            "label_entropy": h_label,
            "label_mi_frac": mi / max(h_label, 1e-9),
        }
    )
    return base
