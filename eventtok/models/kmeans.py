"""k-means over action chunks — the baseline the learned tokenizer has to beat.

Measured on 40 SwingXtimes episodes this outperformed the neural tokenizer on
every metric that matters, so it is a first-class path rather than a diagnostic:

                    within-event change   boundary P   label MI   n-gram==N
    k-means K=32           11.6%             0.127       59.5%     29/40
    neural (4 epochs)      21.4%             0.061       37.2%      0/20

Nothing here is clever. It is Lloyd's algorithm on normalised delta-action chunks,
and it produces the invariance the design needs: both right-side visits of a swing
land on one centroid, both left-side visits on another. That is the property the
learned version keeps failing to reach, and it is why the action trajectory rather
than vision defines the event.

Keep this in the comparison for every future result. If a learned tokenizer cannot
beat Lloyd's algorithm, the learning is not earning its place.
"""

from __future__ import annotations

import json

import numpy as np

from ..data.index import Episode
from ..data.meta import TaskMeta


class KMeansTokenizer:
    def __init__(self, n_clusters: int = 32, seed: int = 0) -> None:
        self.n_clusters = n_clusters
        self.seed = seed
        self.centroids: np.ndarray | None = None
        self.action_scale: np.ndarray | None = None

    # ------------------------------------------------------------------ fit

    def _chunks(self, meta: TaskMeta, episodes: list[Episode]) -> np.ndarray:
        rows = []
        for ep in episodes:
            lo, hi = meta.rows(ep.epis_idx)
            rows.extend(range(lo, hi))
        scale = meta.action_scale
        return np.stack(
            [(meta.delta_actions(r) / scale).ravel() for r in rows]
        ).astype(np.float32)

    def fit(self, meta: TaskMeta, episodes: list[Episode]) -> "KMeansTokenizer":
        from scipy.cluster.vq import kmeans2

        X = self._chunks(meta, episodes)
        np.random.seed(self.seed)
        centroids, _ = kmeans2(X, self.n_clusters, minit="++", seed=self.seed)
        self.centroids = centroids
        self.action_scale = meta.action_scale
        return self

    # ------------------------------------------------------------------ encode

    def encode_chunks(self, chunks: np.ndarray) -> np.ndarray:
        """``(n, k, action_dim)`` already normalised -> cluster ids ``(n,)``."""
        if self.centroids is None:
            raise RuntimeError("fit() first")
        flat = chunks.reshape(len(chunks), -1).astype(np.float32)
        d = ((flat[:, None, :] - self.centroids[None]) ** 2).sum(-1)
        return d.argmin(1)

    def stream_for_episode(self, meta: TaskMeta, ep: Episode) -> list[int]:
        lo, hi = meta.rows(ep.epis_idx)
        scale = self.action_scale if self.action_scale is not None else meta.action_scale
        chunks = np.stack([meta.delta_actions(r) / scale for r in range(lo, hi)])
        return self.encode_chunks(chunks).tolist()

    # ------------------------------------------------------------------ io

    def save(self, path) -> None:
        np.savez(
            str(path),
            centroids=self.centroids,
            action_scale=self.action_scale,
            n_clusters=self.n_clusters,
            seed=self.seed,
        )

    @classmethod
    def load(cls, path) -> "KMeansTokenizer":
        with np.load(str(path)) as d:
            obj = cls(int(d["n_clusters"]), int(d["seed"]))
            obj.centroids = d["centroids"]
            obj.action_scale = d["action_scale"]
        return obj
