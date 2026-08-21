"""Repack per-frame visual features into one contiguous array per episode.

``features/episode_{e}/token_emb_{t}.npy`` is one ~600 KB pickled dict per frame,
768,897 files totalling 432 GiB. Opening one file per sample would make netscratch
latency dominate training, so each task is repacked once into per-episode arrays.

Only ``image_emb_2x2`` (4 tokens x 2048) is kept by default: 16 KB/frame, so
SwingXtimes lands at ~712 MB and PickXtimes at ~880 MB, both RAM-resident. Pass
``scale="4x4"`` if 4 tokens prove too coarse for event identity.

The stored features are raw SigLIP activations (absmax ~130); normalisation is
left to training so the cache stays a faithful copy.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import paths
from .index import Episode, RoboMMEIndex

SCALES = ("2x2", "4x4", "8x8")
_N_TOKENS = {"2x2": 4, "4x4": 16, "8x8": 64}


def cache_dir(task: str, scale: str) -> "paths.Path":
    return paths.CACHE_ROOT / "feats" / f"{task}_{scale}"


def cache_path(task: str, scale: str, epis_idx: int) -> "paths.Path":
    return cache_dir(task, scale) / f"episode_{epis_idx}.npy"


def _read_frame(epis_idx: int, t: int, key: str) -> np.ndarray:
    path = paths.FEATURE_DIR / f"episode_{epis_idx}" / f"token_emb_{t}.npy"
    record = np.load(path, allow_pickle=True).item()
    # bfloat16 -> float32 -> float16. Going straight to fp16 from bfloat16 is
    # not a supported numpy cast.
    return np.asarray(record[key], dtype=np.float32).astype(np.float16)[0]


def repack_episode(
    ep: Episode,
    scale: str = "2x2",
    workers: int = 32,
    overwrite: bool = False,
) -> "paths.Path":
    """Write ``(T, n_tokens, 2048)`` fp16 for one episode. Returns the path.

    Indexed by **absolute** frame ``t`` over ``range(ep.n_frames)``, matching how
    the source files are named, so callers index with ``step_idx`` directly and
    do not need to offset by ``exec_start``.
    """
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}, got {scale!r}")
    out = cache_path(ep.task, scale, ep.epis_idx)
    if out.is_file() and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    key = f"image_emb_{scale}"
    frames = range(ep.n_frames)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        stacked = list(pool.map(lambda t: _read_frame(ep.epis_idx, t, key), frames))

    array = np.stack(stacked, axis=0)
    expected = (ep.n_frames, _N_TOKENS[scale], 2048)
    if array.shape != expected:
        raise ValueError(f"episode {ep.epis_idx}: got {array.shape}, want {expected}")

    # Write via a handle: np.save() silently appends ".npy" to a path that does
    # not already end in it, which would leave the temp file misnamed.
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, array)
    tmp.rename(out)
    return out


def repack_task(
    task: str,
    scale: str = "2x2",
    workers: int = 32,
    overwrite: bool = False,
    index: RoboMMEIndex | None = None,
    progress: bool = True,
):
    index = index or RoboMMEIndex()
    episodes = index.by_task(task)
    written = []
    for i, ep in enumerate(episodes):
        path = repack_episode(ep, scale=scale, workers=workers, overwrite=overwrite)
        written.append(path)
        if progress:
            print(
                f"[{i + 1:3d}/{len(episodes)}] ep{ep.epis_idx} "
                f"T={ep.n_frames:4d} -> {path.name}",
                flush=True,
            )
    return written


class EpisodeFeatures:
    """Lazy per-episode view over the repacked cache."""

    def __init__(self, task: str, scale: str = "2x2") -> None:
        self.task = task
        self.scale = scale
        self._cache: dict[int, np.ndarray] = {}

    def __getitem__(self, epis_idx: int) -> np.ndarray:
        if epis_idx not in self._cache:
            path = cache_path(self.task, self.scale, epis_idx)
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} missing; run scripts/repack_task.py --task {self.task}"
                )
            self._cache[epis_idx] = np.load(path, mmap_mode="r")
        return self._cache[epis_idx]
