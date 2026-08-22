"""Ground-truth event boundaries and labels from the ``simple_subgoal`` field.

RoboMME annotates every timestep with a subgoal string that changes exactly at
event transitions, which gives boundaries and labels for free. Verified on
episode 235 (472 frames, ButtonUnmaskSwap)::

    t=  0   press the first button
    t=117   press the second button
    t=213   pick up the container that hides the red cube
    t=322   put down the container
    t=363   pick up the container that hides the green cube

**These annotations are for evaluation and naming only, never for training.**
Keeping them out is what makes the resulting codes task-agnostic, which the
transfer claim depends on; "we never used the labels" is also a stronger result.
This module is deliberately separate from ``robomme.py`` so the training path
cannot import it by accident.

The strings live only inside the pkls, so extraction reads every pkl of an
episode once (~400 KB each) and caches the result.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass

import numpy as np

from .. import paths
from .index import Episode, RoboMMEIndex


@dataclass(frozen=True)
class Segment:
    start: int      # absolute frame index, inclusive
    end: int        # absolute frame index, exclusive
    label: str      # the simple_subgoal string held over this span

    @property
    def length(self) -> int:
        return self.end - self.start


def _as_scalar(value):
    array = np.asarray(value)
    return array.reshape(-1)[0].item() if array.size else None


def _cache_path(task: str) -> "paths.Path":
    return paths.CACHE_ROOT / "subgoals" / f"{task}.json"


def read_subgoal_track(ep: Episode, field: str = "simple_subgoal") -> list[str]:
    """The per-frame subgoal string for one episode's execution frames."""
    track: list[str] = []
    for pkl_id in range(ep.pkl_lo, ep.pkl_hi + 1):
        with open(paths.PKL_DIR / f"{pkl_id}.pkl", "rb") as fh:
            record = pickle.load(fh)
        track.append(str(record[field]))
    return track


def segments_from_track(track: list[str], exec_start: int) -> list[Segment]:
    """Collapse a per-frame label track into contiguous segments."""
    if not track:
        return []
    out: list[Segment] = []
    start = 0
    for i in range(1, len(track)):
        if track[i] != track[i - 1]:
            out.append(Segment(exec_start + start, exec_start + i, track[start]))
            start = i
    out.append(Segment(exec_start + start, exec_start + len(track), track[start]))
    return out


def extract_task(task: str, index: RoboMMEIndex | None = None, use_cache: bool = True):
    """Boundaries and labels for every episode of ``task``.

    Returns ``{epis_idx: [Segment, ...]}``. Cached to disk on first run, since
    extraction reads every pkl of the task.
    """
    index = index or RoboMMEIndex()
    cache = _cache_path(task)
    if use_cache and cache.is_file():
        with open(cache) as fh:
            raw = json.load(fh)
        return {
            int(k): [Segment(s["start"], s["end"], s["label"]) for s in v]
            for k, v in raw.items()
        }

    result: dict[int, list[Segment]] = {}
    for ep in index.by_task(task):
        track = read_subgoal_track(ep)
        result[ep.epis_idx] = segments_from_track(track, ep.exec_start)

    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w") as fh:
            json.dump(
                {
                    str(k): [
                        {"start": s.start, "end": s.end, "label": s.label} for s in v
                    ]
                    for k, v in result.items()
                },
                fh,
            )
    return result


_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}
_ORDINAL_SUFFIX = re.compile(
    r"\s+for the (" + "|".join(_ORDINALS) + r") time\s*$", re.IGNORECASE
)


def canonical_label(label: str) -> str:
    """Strip the repetition ordinal, so repeats share one label.

    RoboMME annotates repeated actions with an ordinal — "move to the top of the
    right-side target **for the second time**". Physically identical repetitions
    therefore carry different strings.

    This matters for evaluation. A correctly-working tokenizer assigns the *same*
    code to every repetition, so scoring code consistency against the raw labels
    would count that as a failure. Canonicalise before measuring agreement, and
    read the count off :func:`ordinal` instead.
    """
    return _ORDINAL_SUFFIX.sub("", label).strip()


def ordinal(label: str) -> int | None:
    """Which repetition this segment is, from the label's ordinal, else None."""
    match = _ORDINAL_SUFFIX.search(label)
    return _ORDINALS[match.group(1).lower()] if match else None


_OBJECT = re.compile(r"\b(red|green|blue|yellow|purple|orange)\b", re.IGNORECASE)


def observable_label(label: str) -> str:
    """Canonical label with object identity collapsed to ``<obj>``.

    **Use this, not the raw label, when scoring what a per-transition code can
    know.** RoboMME's subgoals name the object — "pick up the container that hides
    the *red* cube" — but on occlusion tasks that object is *hidden by the
    container* at pick time and the colour varies only across episodes (ButtonUnmask:
    red 0.50/ep, green 0.30, blue 0.45, at most one pick each). No code computed
    from a single transition can carry it, so scoring against the raw label measures
    an impossible target and makes a working tokenizer look broken.

    Concretely, on ButtonUnmask this changed the multimodal reading from 38.0% to
    53.9% of label entropy, and flipped the conclusion from "vision does not help
    here" to "vision adds 5.4 points".

    The object binding is not lost — supplying it is precisely the memory log's job.
    Event tokens carry *observable* events; which cube went under which container is
    recovered from earlier entries in the log.
    """
    return _OBJECT.sub("<obj>", canonical_label(label)).strip()


def event_counts(segments: list[Segment]) -> dict[str, int]:
    """How many times each canonical event occurs — per-event count ground truth.

    Finer-grained than the prompt regex, which gives one N for the whole episode.
    Use this as the target for the counting probe and as a cross-check on
    :func:`eventtok.data.prompts.extract_count`.
    """
    counts: dict[str, int] = {}
    for s in segments:
        key = canonical_label(s.label)
        counts[key] = counts.get(key, 0) + 1
    return counts


def boundaries(segments: list[Segment]) -> list[int]:
    """Start frames of each segment — the event boundaries."""
    return [s.start for s in segments]


def vocabulary(per_episode: dict[int, list[Segment]]) -> dict[str, int]:
    """Distinct subgoal strings across a task, with occurrence counts."""
    counts: dict[str, int] = {}
    for segs in per_episode.values():
        for s in segs:
            counts[s.label] = counts.get(s.label, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
