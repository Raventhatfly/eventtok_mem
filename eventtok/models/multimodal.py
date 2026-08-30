"""Event tokens from action **and** vision. The path the log is supposed to use.

``KMeansTokenizer`` is action-only by design -- it is the OAT-shaped control the
multimodal result is measured against. It is also the convenient path, and that is why
this project has drifted back to it twice: every log-based experiment so far
(consumption probe, diffusion policy, the rollout design) built its code stream with it,
so the tokens the policy read were action-only tokens and the whole memory story was
being told with OAT tokens.

This class makes the multimodal path as easy to call, so the drift stops being the
default. It is the same construction the 16-task sweep used for label accuracy --
centred, energy-normalised blocks, vision through PCA -- applied to produce a code
stream that the BPE and log stages consume.

Why the normalisation is not optional here: the shared mean is 99.8% of SigLIP's
feature energy and 94.9% of DINOv2's, so Euclidean k-means on raw features clusters a
constant. Blocks are also scaled to equal total variance so a 4096-dimensional vision
block does not swamp a 160-dimensional action block by dimension count alone.
"""

from __future__ import annotations

import numpy as np

from ..data.index import Episode
from ..data.meta import TaskMeta


class MultimodalTokenizer:
    """k-means over normalised action chunks concatenated with visual context.

    Args:
        n_clusters: codebook size.
        horizon: frames between ``feat_t`` and ``feat_next``; the visual delta is the
            informative part, since the state alone barely moves over the horizon.
        pca: vision components retained. Fixed across encoders so a comparison is
            about the encoder rather than its width.
        vision_weight: relative variance given to the vision block. 1.0 means action
            and vision contribute equally. Lower it when vision carries no signal --
            on PatternLock, vision alone scores *below* its majority baseline, and
            equal weighting there spends half the distance budget on noise.
    """

    def __init__(
        self,
        n_clusters: int = 16,
        horizon: int = 20,
        pca: int = 64,
        vision_weight: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.n_clusters = n_clusters
        self.horizon = horizon
        self.pca = pca
        self.vision_weight = vision_weight
        self.seed = seed
        self.centroids: np.ndarray | None = None
        self._blocks: dict = {}
        self.action_scale: np.ndarray | None = None

    # ------------------------------------------------------------------ build
    def _rows(self, meta: TaskMeta, episodes):
        rows, epis, frame, exec_start = [], [], [], []
        for ep in episodes:
            lo, hi = meta.rows(ep.epis_idx)
            rows.extend(range(lo, hi))
            epis.extend([ep.epis_idx] * (hi - lo))
            frame.extend(range(hi - lo))
            exec_start.extend([ep.exec_start] * (hi - lo))
        return (np.asarray(rows), np.asarray(epis), np.asarray(frame),
                np.asarray(exec_start))

    def _features(self, meta: TaskMeta, getter, rows, epis, frame, exec_start):
        from ..scripts.compare_modalities import action_matrix, vision_matrix

        raw = {"action": action_matrix(meta, rows)}
        if getter is not None:
            offsets = (
                exec_start if getattr(getter, "indexes_absolute_frames", True)
                else np.zeros_like(exec_start)
            )
            raw["vision"] = vision_matrix(
                getter, epis, frame, self.horizon, "both", offsets
            )
        return raw

    def fit(self, meta: TaskMeta, episodes, getter=None) -> "MultimodalTokenizer":
        from scipy.cluster.vq import kmeans2

        from ..scripts.compare_modalities import Block

        rows, epis, frame, exec_start = self._rows(meta, episodes)
        raw = self._features(meta, getter, rows, epis, frame, exec_start)
        self._blocks = {
            name: Block(name, None if name == "action" else self.pca).fit(
                X, seed=self.seed
            )
            for name, X in raw.items()
        }
        X = self._stack(raw)
        np.random.seed(self.seed)
        self.centroids, _ = kmeans2(X, self.n_clusters, minit="++", seed=self.seed)
        self.action_scale = meta.action_scale
        self.has_vision = "vision" in raw
        return self

    def raw_for(self, meta: TaskMeta, episodes, getter=None) -> dict:
        """Un-normalised feature blocks for these episodes. Exposed for a joint fit."""
        return self._features(meta, getter, *self._rows(meta, episodes))

    def fit_raw(self, raws: list[dict], max_rows: int | None = None) -> "MultimodalTokenizer":
        """Fit blocks and centroids on feature matrices pooled across several tasks.

        One policy trained on 16 tasks needs one vocabulary: with a per-task fit, token 7
        means a different motion in each task and its embedding row is asked to be all of
        them at once.

        Rows are subsampled to ``max_rows`` before stacking. The raw vision block is
        16k-dimensional, so all 16 tasks at full length is ~26 GB, while k-means at k=16
        and a 64-component PCA are both well determined by a few 10k rows.
        """
        from scipy.cluster.vq import kmeans2

        from ..scripts.compare_modalities import Block

        names = list(raws[0])
        for r in raws:
            if list(r) != names:
                raise ValueError(f"blocks differ across tasks: {names} vs {list(r)}")

        rng = np.random.default_rng(self.seed)
        if max_rows is not None:
            per = max(1, max_rows // len(raws))
            keep = [
                rng.choice(len(r[names[0]]), min(per, len(r[names[0]])), replace=False)
                for r in raws
            ]
            raws = [{n: r[n][k] for n in names} for r, k in zip(raws, keep)]

        pooled = {n: np.concatenate([r[n] for r in raws], axis=0) for n in names}
        self._blocks = {
            n: Block(n, None if n == "action" else self.pca).fit(pooled[n], seed=self.seed)
            for n in names
        }
        X = self._stack(pooled)
        np.random.seed(self.seed)
        self.centroids, _ = kmeans2(X, self.n_clusters, minit="++", seed=self.seed)
        self.has_vision = "vision" in names
        return self

    def encode_raw(self, raw: dict) -> np.ndarray:
        """Nearest centroid per row, for an already-built raw block dict."""
        if self.centroids is None:
            raise RuntimeError("fit() first")
        X = self._stack(raw)
        return ((X[:, None, :] - self.centroids[None]) ** 2).sum(-1).argmin(1)

    def _stack(self, raw) -> np.ndarray:
        parts = []
        for name, X in raw.items():
            Z = self._blocks[name].transform(X)
            if name == "vision":
                Z = Z * float(self.vision_weight)
            parts.append(Z)
        return np.concatenate(parts, axis=1).astype(np.float32)

    # ------------------------------------------------------------------ encode
    def stream_for_episodes(self, meta: TaskMeta, episodes, getter=None) -> dict:
        """``{epis_idx: [code, ...]}`` for the given episodes."""
        if self.centroids is None:
            raise RuntimeError("fit() first")
        rows, epis, frame, exec_start = self._rows(meta, episodes)
        raw = self._features(meta, getter, rows, epis, frame, exec_start)
        X = self._stack(raw)
        d = ((X[:, None, :] - self.centroids[None]) ** 2).sum(-1)
        codes = d.argmin(1)
        out, i = {}, 0
        for ep in episodes:
            lo, hi = meta.rows(ep.epis_idx)
            n = hi - lo
            out[ep.epis_idx] = codes[i : i + n].tolist()
            i += n
        return out
