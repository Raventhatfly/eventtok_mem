"""Controls that could make the counting result meaningless. Run these first.

The counting number (96% on unseen N) came from k-means codes plus a pattern. That is
only a result if something simpler does not do as well. Three candidates, each fitted
under the *same* protocol as the tokenizer -- parameters chosen on the N in {1,2,3}
fit episodes only, then applied unchanged to held-out episodes:

``length``   Predict N from episode length by least squares, then round. Reported
             first because ``corr(episode_length, N) = +0.977`` on PickXtimes and the
             per-N length ranges barely overlap. Doing a task N times takes N times as
             long; if that is all the counting result recovers, the tokenizer is
             restating the task structure.
``gripper``  Count gripper open/close cycles. One pick is one grasp, so on PickXtimes
             this is a direct count and needs no representation at all.
``peaks``    Count peaks in the delta-action magnitude -- a generic periodicity
             detector over the trajectory, with no notion of an event.

A baseline that matches the tokenizer does not mean the tokenizer is broken; it means
the *counting benchmark* does not distinguish them, and the claim has to move to
something that does.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..data.index import Episode
from ..data.meta import TaskMeta


def _fit_round_linear(x: np.ndarray, n: np.ndarray):
    """Least-squares N ~ a*x + b, returned as a rounding predictor."""
    a, b = np.polyfit(x, n, 1)
    return lambda v: int(np.clip(round(float(a * v + b)), 1, 99))


def length_feature(meta: TaskMeta, ep: Episode) -> float:
    lo, hi = meta.rows(ep.epis_idx)
    return float(hi - lo)


def gripper_cycles(meta: TaskMeta, ep: Episode, threshold: float, min_gap: int = 15) -> int:
    """Down-crossings of the gripper channel, debounced by ``min_gap`` frames."""
    lo, hi = meta.rows(ep.epis_idx)
    g = np.asarray(meta.state[lo:hi, 7], dtype=np.float64)
    closed = g > threshold
    count, last = 0, -10**9
    for i in range(1, len(closed)):
        if closed[i] and not closed[i - 1] and i - last >= min_gap:
            count += 1
            last = i
    return count


def action_peaks(meta: TaskMeta, ep: Episode, threshold: float, min_gap: int = 25) -> int:
    """Peaks in the normalised delta-action magnitude -- generic periodicity."""
    lo, hi = meta.rows(ep.epis_idx)
    scale = meta.action_scale
    mag = np.array(
        [np.abs(meta.delta_actions(r) / scale).mean() for r in range(lo, hi)]
    )
    if len(mag) < 3:
        return 0
    count, last = 0, -10**9
    for i in range(1, len(mag) - 1):
        if mag[i] >= mag[i - 1] and mag[i] > mag[i + 1] and mag[i] > threshold:
            if i - last >= min_gap:
                count += 1
                last = i
    return count


def fit_predictors(
    meta: TaskMeta, fit_eps: Sequence[Episode], counts: dict[int, int]
) -> dict:
    """Choose each baseline's parameters on the fit episodes only."""
    n = np.array([counts[e.epis_idx] for e in fit_eps], dtype=float)

    lengths = np.array([length_feature(meta, e) for e in fit_eps])
    length_fn = _fit_round_linear(lengths, n)

    def best_threshold(counter, grid):
        best, best_acc = grid[0], -1.0
        for t in grid:
            pred = np.array([counter(meta, e, t) for e in fit_eps], dtype=float)
            acc = float((pred == n).mean())
            if acc > best_acc:
                best, best_acc = t, acc
        return best, best_acc

    g_lo = float(np.percentile([meta.state[slice(*meta.rows(e.epis_idx))][:, 7].max()
                                for e in fit_eps], 5))
    g_thresh, g_acc = best_threshold(
        gripper_cycles, list(np.linspace(1e-4, max(g_lo, 1e-3), 12))
    )
    p_thresh, p_acc = best_threshold(
        action_peaks, list(np.linspace(0.05, 1.5, 12))
    )
    return {
        "length": {"predict": lambda e: length_fn(length_feature(meta, e)),
                   "param": None, "fit_acc": None},
        "gripper": {"predict": lambda e, t=g_thresh: gripper_cycles(meta, e, t),
                    "param": g_thresh, "fit_acc": g_acc},
        "peaks": {"predict": lambda e, t=p_thresh: action_peaks(meta, e, t),
                  "param": p_thresh, "fit_acc": p_acc},
    }
