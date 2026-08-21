"""One pass over a task's pkls, caching everything except the images.

Actions and subgoal labels both live only inside the pkls, and each pkl is
~400 KB dominated by two 256x256 RGB frames. Reading them at training time to
recover a (20, 8) action chunk would be absurd, and reading them separately for
actions and for subgoals would double an already expensive scan. So one pass
extracts both, plus state, into a compact per-task npz.

Sizes for SwingXtimes (43,419 execution frames): actions 28 MB, state 1.4 MB,
labels a few hundred KB. The source pkls are ~17 GB.

Note ``actions`` in a pkl at absolute frame ``t`` is already the 20-step chunk
starting at ``t``, so for the default ``k = 20`` the chunk needs no assembly.
"""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import paths
from .index import Episode, RoboMMEIndex


def cache_path(task: str) -> "paths.Path":
    return paths.CACHE_ROOT / "meta" / f"{task}.npz"


def _read_pkl(pkl_id: int) -> dict:
    with open(paths.PKL_DIR / f"{pkl_id}.pkl", "rb") as fh:
        return pickle.load(fh)


def _as_scalar(value):
    array = np.asarray(value)
    return array.reshape(-1)[0].item() if array.size else None


def build(
    task: str,
    index: RoboMMEIndex | None = None,
    overwrite: bool = False,
    progress: bool = True,
    workers: int = 32,
) -> "paths.Path":
    """Scan every pkl of ``task`` once; write actions, state and labels."""
    out = cache_path(task)
    if out.is_file() and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    index = index or RoboMMEIndex()
    episodes = index.by_task(task)

    epis_idx: list[int] = []
    step_idx: list[int] = []
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    labels: list[str] = []

    # Threaded: netscratch open() latency dominates, not deserialisation.
    # ThreadPoolExecutor.map preserves order, which the row layout relies on.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, ep in enumerate(episodes):
            ids = range(ep.pkl_lo, ep.pkl_hi + 1)
            for record in pool.map(_read_pkl, ids):
                epis_idx.append(int(_as_scalar(record["epis_idx"])))
                step_idx.append(int(_as_scalar(record["step_idx"])))
                actions.append(np.asarray(record["actions"], dtype=np.float32))
                states.append(np.asarray(record["state"], dtype=np.float32))
                labels.append(str(record["simple_subgoal"]))
            if progress:
                print(
                    f"[{i + 1:3d}/{len(episodes)}] ep{ep.epis_idx} ({ep.n_exec} frames)",
                    flush=True,
                )

    # Write via a handle: np.savez_compressed() appends ".npz" to any path that
    # does not already end in it, which would misname the temp file. Same trap
    # as np.save() in repack.py.
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            epis_idx=np.asarray(epis_idx, dtype=np.int32),
            step_idx=np.asarray(step_idx, dtype=np.int32),
            actions=np.stack(actions),          # (n_exec, 20, 8) fp32
            state=np.stack(states),             # (n_exec, 8) fp32
            labels=np.asarray(labels, dtype=object),
        )
    tmp.rename(out)
    return out


class TaskMeta:
    """Read side of the cache, with per-episode slicing."""

    def __init__(self, task: str) -> None:
        self.task = task
        path = cache_path(task)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} missing; run scripts/build_meta.py --task {task}"
            )
        with np.load(path, allow_pickle=True) as data:
            self.epis_idx = data["epis_idx"]
            self.step_idx = data["step_idx"]
            self.actions = data["actions"]
            self.state = data["state"]
            self.labels = data["labels"]

        # Row ranges per episode. epis_idx is non-decreasing by construction.
        self._rows: dict[int, tuple[int, int]] = {}
        changes = np.flatnonzero(np.diff(self.epis_idx)) + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [len(self.epis_idx)]))
        for lo, hi in zip(starts, ends):
            self._rows[int(self.epis_idx[lo])] = (int(lo), int(hi))

    def rows(self, epis_idx: int) -> tuple[int, int]:
        return self._rows[epis_idx]

    def row_of(self, ep: Episode, step_idx: int) -> int:
        lo, _ = self._rows[ep.epis_idx]
        return lo + (step_idx - ep.exec_start)

    def episode_labels(self, epis_idx: int) -> list[str]:
        lo, hi = self._rows[epis_idx]
        return [str(x) for x in self.labels[lo:hi]]

    def episode_actions(self, epis_idx: int) -> np.ndarray:
        lo, hi = self._rows[epis_idx]
        return self.actions[lo:hi]

    @property
    def action_scale(self) -> np.ndarray:
        """Per-dimension std of the delta action chunks.

        The gripper dimension has std ~0.95 while the seven pose dimensions sit
        at 0.05-0.20, so an unnormalised L1 is effectively a gripper predictor
        and the pose information is invisible to the loss. Computed once over a
        subsample and cached on the instance.
        """
        if getattr(self, "_action_scale", None) is None:
            rows = np.linspace(0, len(self.actions) - 1, num=min(4096, len(self.actions)))
            rows = np.unique(rows.astype(np.int64))
            chunks = np.stack([self.delta_actions(int(r)) for r in rows])
            scale = chunks.reshape(-1, chunks.shape[-1]).std(axis=0)
            self._action_scale = np.maximum(scale, 1e-3).astype(np.float32)
        return self._action_scale

    def delta_actions(self, row: int) -> np.ndarray:
        """Action chunk with the current state removed from the first 7 dims.

        Absolute end-effector targets make the same motion toward two different
        object placements look unrelated, which would fragment codes by location.
        The gripper dimension stays absolute. Mirrors OpenPI's DeltaActions.
        """
        chunk = self.actions[row].copy()
        chunk[:, :7] -= self.state[row][:7]
        return chunk
