"""Episode/timestep index over the RoboMME preprocessed pkls.

The dataset ships its own index, so nothing here scans the 476,857 pkl files.
``robomme_data_h5/fluxvla_hdf5_index.json`` (10 MB) contains

    episodes : 1600 dicts {path, episode_id, task, timestep_ids}
    index    : 476,857 pairs [epis_idx, step_idx], one per pkl, in pkl order

so ``index[i]`` identifies ``data/{i}.pkl`` directly.

Two facts about the layout that are easy to get wrong:

* ``data/`` holds **execution frames only**, concatenated in ``epis_idx`` order.
  So ``step_idx`` is the absolute frame index in the episode and starts at
  ``exec_start_idx``, which is > 0 for 900 of the 1600 episodes. Both counting
  tasks happen to have ``exec_start == 0`` throughout, but code must not assume it.
* ``features/episode_{e}/token_emb_{t}.npy`` is indexed by the **absolute** frame
  ``t`` and exists for all frames including the pre-execution prefix. Joining
  features to pkls therefore goes through ``step_idx``, not through a 0-based
  position within the pkl range.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .. import paths
from . import prompts

# Alphabetical, matching epis_idx // 100. Verified against the index json for
# all 1600 episodes.
TASKS = (
    "BinFill",
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    "InsertPeg",
    "MoveCube",
    "PatternLock",
    "PickHighlight",
    "PickXtimes",
    "RouteStick",
    "StopCube",
    "SwingXtimes",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "VideoRepick",
    "VideoUnmask",
    "VideoUnmaskSwap",
)

EPISODES_PER_TASK = 100


@dataclass(frozen=True)
class Episode:
    epis_idx: int           # global 0..1599
    task: str               # e.g. "SwingXtimes"
    episode_id: int         # 0..99 within the task
    prompt: str
    n_frames: int           # T_e, all frames including any pre-execution prefix
    exec_start: int         # first step_idx present in data/
    pkl_lo: int             # first pkl id for this episode (inclusive)
    pkl_hi: int             # last pkl id for this episode (inclusive)
    count: int | None       # target repetition N, for counting tasks only

    @property
    def n_exec(self) -> int:
        return self.pkl_hi - self.pkl_lo + 1

    def pkl_id(self, step_idx: int) -> int:
        """pkl id for an absolute frame index."""
        if not self.exec_start <= step_idx < self.n_frames:
            raise IndexError(
                f"step {step_idx} outside execution range "
                f"[{self.exec_start}, {self.n_frames}) of episode {self.epis_idx}"
            )
        return self.pkl_lo + (step_idx - self.exec_start)


class RoboMMEIndex:
    """Loads the shipped index and derives per-episode pkl ranges."""

    def __init__(self, index_json=None) -> None:
        self.index_json = index_json or paths.INDEX_JSON
        with open(self.index_json) as fh:
            raw = json.load(fh)

        self._raw_episodes = raw["episodes"]
        # (N, 2) of [epis_idx, step_idx], in pkl order.
        self.pairs = np.asarray(raw["index"], dtype=np.int64)
        if self.pairs.ndim != 2 or self.pairs.shape[1] != 2:
            raise ValueError(f"unexpected index shape {self.pairs.shape}")

        self._episodes = self._build_episodes()

    def _build_episodes(self) -> list[Episode]:
        epis = self.pairs[:, 0]
        steps = self.pairs[:, 1]

        # epis_idx is non-decreasing in pkl order, so episode boundaries are
        # exactly the positions where it changes.
        if np.any(np.diff(epis) < 0):
            raise ValueError(
                "index is not sorted by epis_idx; the per-episode pkl ranges "
                "below assume it is"
            )
        starts = np.concatenate(([0], np.flatnonzero(np.diff(epis)) + 1))
        ends = np.concatenate((starts[1:] - 1, [len(epis) - 1]))

        out: list[Episode] = []
        for lo, hi in zip(starts, ends):
            e = int(epis[lo])
            meta = self._raw_episodes[e]
            task = TASKS[e // EPISODES_PER_TASK]
            prompt = meta["task"]
            out.append(
                Episode(
                    epis_idx=e,
                    task=task,
                    episode_id=int(meta["episode_id"]),
                    prompt=prompt,
                    n_frames=len(meta["timestep_ids"]),
                    exec_start=int(steps[lo]),
                    pkl_lo=int(lo),
                    pkl_hi=int(hi),
                    count=prompts.extract_count(task, prompt),
                )
            )
        return out

    # ------------------------------------------------------------------ access

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, epis_idx: int) -> Episode:
        ep = self._episodes[epis_idx]
        if ep.epis_idx != epis_idx:
            # Only possible if some episode contributed zero pkls.
            raise LookupError(f"episode {epis_idx} missing from the index")
        return ep

    @cached_property
    def episodes(self) -> tuple[Episode, ...]:
        return tuple(self._episodes)

    def task_of(self, epis_idx: int) -> str:
        """Task name from a global episode index, without touching the index file.

        ``epis_idx // 100`` is the position in the alphabetical task list. Cheap enough
        to call per sample in a data loader, which ``self[epis_idx].task`` is not.
        """
        return TASKS[int(epis_idx) // EPISODES_PER_TASK]

    def by_task(self, task: str) -> list[Episode]:
        if task not in TASKS:
            raise KeyError(f"unknown task {task!r}; expected one of {TASKS}")
        return [ep for ep in self._episodes if ep.task == task]

    def by_count(self, task: str, counts) -> list[Episode]:
        """Episodes of ``task`` whose target N is in ``counts``.

        This is the extrapolation split: train on {1,2,3}, hold out {4,5}.
        """
        wanted = set(counts)
        return [ep for ep in self.by_task(task) if ep.count in wanted]

    def locate(self, pkl_id: int) -> tuple[int, int]:
        """(epis_idx, step_idx) for a pkl id."""
        return tuple(int(v) for v in self.pairs[pkl_id])  # type: ignore[return-value]
