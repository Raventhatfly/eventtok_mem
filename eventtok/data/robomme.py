"""Transition dataset for the event tokenizer.

One item is a single transition over ``k`` frames::

    feat_t      (n_tokens, 2048)   visual features at the start
    feat_next   (n_tokens, 2048)   visual features at t + k
    actions     (k, 8)             delta action chunk over the interval

The tokenizer quantises the *transition*, not the frame: the decoder is given
``feat_t`` for free, so the code only has to carry what changed. That is what
makes small codebooks come out meaning "reach", "grasp", "lift" rather than
describing the scene.

``k = 20`` at RoboMME's 20 Hz is one second. Verified against episode 235, whose
annotated events run 41-117 frames, so an event spans 2-6 transitions — enough
repetition for the BPE stage to find a pattern, which one code per event would
not give. It also matches LAPA (~0.6 s) and UniVLA (~1 s).

Features are stored raw (SigLIP activations, absmax ~130). Normalisation is a
LayerNorm at the encoder input rather than a statistics file, so the cache stays
a faithful copy and there is one less artifact to keep in sync.

This module deliberately does **not** import ``subgoals``. The annotations are
for evaluation and naming only; keeping the training path unable to reach them
is what makes "we never used the labels" structurally true rather than a promise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .index import Episode, RoboMMEIndex
from .meta import TaskMeta
from .repack import EpisodeFeatures


@dataclass(frozen=True)
class TransitionRef:
    epis_idx: int
    step: int          # absolute frame index of the transition start
    row: int           # row into TaskMeta arrays


class TransitionDataset(Dataset):
    """Transitions over one task, or over an explicit episode subset."""

    def __init__(
        self,
        task: str,
        k: int = 20,
        scale: str = "2x2",
        stride: int = 1,
        episodes: list[Episode] | None = None,
        index: RoboMMEIndex | None = None,
        delta_actions: bool = True,
        normalize_actions: bool = True,
    ) -> None:
        self.task = task
        self.k = k
        self.scale = scale
        self.delta_actions = delta_actions
        self.normalize_actions = normalize_actions

        self.index = index or RoboMMEIndex()
        self.meta = TaskMeta(task)
        self.features = EpisodeFeatures(task, scale)
        self.episodes = episodes if episodes is not None else self.index.by_task(task)

        self.refs: list[TransitionRef] = []
        for ep in self.episodes:
            lo, hi = self.meta.rows(ep.epis_idx)
            # Need feat at t and t+k, so t+k must be a real frame; and the action
            # chunk must exist, so t must be an execution frame.
            last = min(ep.n_frames - 1 - k, ep.exec_start + (hi - lo) - 1)
            for step in range(ep.exec_start, last + 1, stride):
                self.refs.append(
                    TransitionRef(ep.epis_idx, step, self.meta.row_of(ep, step))
                )

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ref = self.refs[i]
        feats = self.features[ref.epis_idx]
        feat_t = np.asarray(feats[ref.step], dtype=np.float32)
        feat_next = np.asarray(feats[ref.step + self.k], dtype=np.float32)

        if self.delta_actions:
            actions = self.meta.delta_actions(ref.row)
        else:
            actions = self.meta.actions[ref.row]
        actions = np.asarray(actions, dtype=np.float32)[: self.k]
        if self.normalize_actions:
            actions = actions / self.meta.action_scale

        return {
            "feat_t": torch.from_numpy(feat_t),
            "feat_next": torch.from_numpy(feat_next),
            "actions": torch.from_numpy(actions),
            "epis_idx": torch.tensor(ref.epis_idx, dtype=torch.long),
            "step": torch.tensor(ref.step, dtype=torch.long),
        }

    # ------------------------------------------------------------------ helpers

    def episode_transitions(self, epis_idx: int) -> list[int]:
        """Dataset positions for one episode, in temporal order.

        Used when tokenising a whole episode into a code stream, where order
        matters and shuffling does not apply.
        """
        return [i for i, r in enumerate(self.refs) if r.epis_idx == epis_idx]


def split_by_count(
    task: str,
    train_counts=(1, 2, 3),
    test_counts=(4, 5),
    index: RoboMMEIndex | None = None,
    **kwargs,
) -> tuple[TransitionDataset, TransitionDataset]:
    """The extrapolation split: train on low N, hold out high N.

    Only PickXtimes supports this — SwingXtimes has no episodes above N = 3, so
    asking it for a held-out {4, 5} set yields an empty dataset.
    """
    index = index or RoboMMEIndex()
    train_eps = index.by_count(task, train_counts)
    test_eps = index.by_count(task, test_counts)
    if not test_eps:
        raise ValueError(
            f"{task} has no episodes with N in {tuple(test_counts)}; "
            f"the extrapolation experiment needs PickXtimes"
        )
    return (
        TransitionDataset(task, episodes=train_eps, index=index, **kwargs),
        TransitionDataset(task, episodes=test_eps, index=index, **kwargs),
    )
