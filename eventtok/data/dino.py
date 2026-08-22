"""DINOv2 features from the raw pkl images, in the same layout as the SigLIP cache.

Why this exists: every vision number so far came from one encoder. The cached
``image_emb_*`` features are SigLIP-So400m/14 projected to pi0.5's 2048-wide token
space -- language-aligned and semantic. OpenVLA pairs SigLIP with DINOv2 precisely
because DINOv2 carries spatial and geometric detail that a language-aligned tower
drops, which is the kind of information an event boundary depends on.

**The comparison is only worth anything if the encoder is the only thing that
changes.** So this reproduces the SigLIP pipeline exactly, verified by reading
``mme_vla_suite/shared/mem_buffer.py:add_buffer`` and ``siglip_tokenizer.py``:

    * one camera. ``add_buffer(image[None, None, ...], ...)`` -- ``num_views=1``,
      and it is the third-person ``image``, never ``wrist_image``.
    * 224x224. The source images are 256x256, so ``resize_with_pad`` is a plain
      bilinear resize with no padding.
    * mean pooling of the patch grid, row-major, non-overlapping, pool size
      ``sqrt(p / target)`` -- ``data_utils.pool_tokens_to_size``.
    * fp16 ``(T, n_tokens, D)`` per episode, indexed by absolute frame.

Two things deliberately differ, both because matching them would be wrong:

    * normalisation. SigLIP wants ``[-1, 1]``, DINOv2 wants ImageNet mean/std.
      Each encoder gets its own correct preprocessing; forcing one to use the
      other's would handicap it.
    * width. DINOv2-large is 1024, the cached SigLIP is 2048. Downstream both go
      through the tokenizer's input LayerNorm and a per-source linear, so width is
      not itself a confound -- but SigLIP's extra width comes from pi0.5's learned
      projector, so those features have seen robot data and DINOv2's have not.
      That favours SigLIP, and is worth saying out loud when reporting the result.

The wrist camera is extracted too, into its own cache directory. Reading the pkls
costs ~28 GB per task and the GPU pass is minutes, so the second camera is nearly
free in the same pass -- but it is a *separate variable* from the encoder swap and
must never be folded into a "DINO vs SigLIP" row.
"""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import paths
from .index import Episode

MODELS = {
    "dinov2l": ("facebook/dinov2-large", 1024),
    "dinov2b": ("facebook/dinov2-base", 768),
}
CAMERAS = ("image", "wrist_image")
_N_TOKENS = {"2x2": 4, "4x4": 16, "8x8": 64}

# DINOv2 was trained at 224 with patch 14, so the grid is 16x16 = 256 patches --
# divisible by every target above, which is what makes the pooling exact.
RESOLUTION = 224
PATCH = 14


def cache_dir(task: str, model: str, scale: str, camera: str) -> "paths.Path":
    cam = "base" if camera == "image" else "wrist"
    return paths.CACHE_ROOT / "feats" / f"{task}_{model}_{cam}_{scale}"


def cache_path(task: str, model: str, scale: str, camera: str, epis_idx: int):
    return cache_dir(task, model, scale, camera) / f"episode_{epis_idx}.npy"


def read_cameras(ep: Episode, cameras=CAMERAS, workers: int = 16) -> dict:
    """``{camera: (T, 256, 256, 3) uint8}`` for one episode, reading each pkl once.

    This is the expensive half of the job: ~400 KB per pkl on netscratch, and a
    pkl load deserialises the whole record whatever you want out of it. Both camera
    images therefore come from one pass -- fetching them separately would double
    34 GB of reads to save nothing. Threads, not processes, since it is I/O wait.
    """
    cameras = tuple(cameras)

    def one(pkl_id: int) -> tuple:
        with open(paths.PKL_DIR / f"{pkl_id}.pkl", "rb") as fh:
            record = pickle.load(fh)
        return tuple(np.asarray(record[c], dtype=np.uint8) for c in cameras)

    ids = range(ep.pkl_lo, ep.pkl_hi + 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, ids))
    return {c: np.stack([r[i] for r in rows], axis=0) for i, c in enumerate(cameras)}


def read_images(ep: Episode, camera: str, workers: int = 16) -> np.ndarray:
    """``(T, 256, 256, 3)`` uint8 for one camera. Convenience wrapper."""
    return read_cameras(ep, (camera,), workers=workers)[camera]


def pool_tokens(tokens, target: int):
    """Mean-pool a patch grid to ``target`` tokens -- SigLIP's rule, in torch.

    ``(B, p, D) -> (B, target, D)``, reshaping ``p`` row-major to ``(h, w)`` and
    average-pooling with non-overlapping windows of ``sqrt(p / target)``. Matching
    this matters: a different pooling would change the comparison from "which
    encoder" to "which encoder and which pooling".
    """
    import torch.nn.functional as F

    b, p, d = tokens.shape
    if p == target:
        return tokens
    h = w = int(round(p**0.5))
    if h * w != p:
        raise ValueError(f"{p} patches is not a square grid")
    size = int(round((p / target) ** 0.5))
    if size * size * target != p:
        raise ValueError(f"cannot pool {p} patches to {target}")
    grid = tokens.reshape(b, h, w, d).permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(grid, kernel_size=size, stride=size)
    return pooled.flatten(2).transpose(1, 2)


class Dinov2Encoder:
    """Frozen DINOv2, batched, fp16 out. Weights must already be cached.

    Compute nodes have no outbound network, so ``HF_HOME`` has to point at a
    populated cache before this runs; ``local_files_only`` turns a silent
    download-hang into an immediate error.
    """

    def __init__(self, model: str = "dinov2l", device=None, batch: int = 128) -> None:
        import torch
        from transformers import Dinov2Model

        if model not in MODELS:
            raise ValueError(f"model must be one of {sorted(MODELS)}, got {model!r}")
        repo_id, self.width = MODELS[model]
        self.name = model
        self.batch = batch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.net = (
            Dinov2Model.from_pretrained(repo_id, local_files_only=True)
            .eval()
            .to(self.device, dtype=torch.float16)
        )
        for p in self.net.parameters():
            p.requires_grad_(False)
        # ImageNet statistics, DINOv2's own preprocessing.
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def encode(self, images: np.ndarray, scale: str = "2x2") -> np.ndarray:
        """``(T, H, W, 3)`` uint8 -> ``(T, n_tokens, width)`` fp16."""
        import torch
        import torch.nn.functional as F

        target = _N_TOKENS[scale]
        out = []
        with torch.inference_mode():
            for lo in range(0, len(images), self.batch):
                chunk = torch.from_numpy(images[lo : lo + self.batch]).to(
                    self.device, non_blocking=True
                )
                x = chunk.permute(0, 3, 1, 2).to(torch.float16) / 255.0
                x = F.interpolate(
                    x, size=(RESOLUTION, RESOLUTION), mode="bilinear", align_corners=False
                )
                x = (x - self.mean.half()) / self.std.half()
                # Drop the CLS token: pooling is over the spatial grid, and CLS
                # has no position in it.
                patches = self.net(pixel_values=x).last_hidden_state[:, 1:]
                expected = (RESOLUTION // PATCH) ** 2
                if patches.shape[1] != expected:
                    raise ValueError(
                        f"{patches.shape[1]} patch tokens, expected {expected}"
                    )
                out.append(pool_tokens(patches, target).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float16)


def encode_episode(
    ep: Episode,
    encoder: Dinov2Encoder,
    scale: str = "2x2",
    cameras=CAMERAS,
    readers: int = 16,
    overwrite: bool = False,
) -> dict:
    """Encode and write one episode per camera. Returns ``{camera: path}``.

    The per-episode file *is* the checkpoint. A preempted job re-runs and skips
    whatever is already on disk, so no separate resume state is needed -- and
    because each file is renamed into place only once complete, a job killed
    mid-write leaves no half-episode behind.
    """
    written = {
        c: cache_path(ep.task, encoder.name, scale, c, ep.epis_idx) for c in cameras
    }
    todo = [c for c in cameras if overwrite or not written[c].is_file()]
    if not todo:
        return written

    # One pkl pass for every camera still needed.
    frames = read_cameras(ep, todo, workers=readers)
    for camera in todo:
        out = written[camera]
        images = frames[camera]
        # n_exec, not n_frames: the pkls hold execution frames only, and the two
        # differ for every episode with a pre-execution prefix.
        if len(images) != ep.n_exec:
            raise ValueError(
                f"episode {ep.epis_idx}: read {len(images)} frames, index says "
                f"{ep.n_exec} execution frames"
            )
        feats = encoder.encode(images, scale=scale)
        expected = (ep.n_exec, _N_TOKENS[scale], encoder.width)
        if feats.shape != expected:
            raise ValueError(f"episode {ep.epis_idx}: {feats.shape} != {expected}")

        out.parent.mkdir(parents=True, exist_ok=True)
        # Write through a handle: np.save() appends ".npy" to any path that does
        # not end in it, which would misname the temp file and break the rename.
        tmp = out.with_name(out.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, feats)
        tmp.rename(out)
    return written


class EpisodeDinoFeatures:
    """Lazy mmap view over the DINO cache.

    ``indexes_absolute_frames`` is False, **unlike** ``repack.EpisodeFeatures``: these
    features are encoded from the pkls, which hold execution frames only, so rows run
    over ``range(ep.n_exec)``. The two conventions coincide on any task with
    ``exec_start == 0`` -- which is every task used so far -- and diverge on the 900
    episodes that have a pre-execution prefix.
    """

    indexes_absolute_frames = False

    def __init__(
        self, task: str, model: str = "dinov2l", scale: str = "2x2", camera: str = "image"
    ) -> None:
        self.task, self.model, self.scale, self.camera = task, model, scale, camera
        self._cache: dict[int, np.ndarray] = {}

    def __getitem__(self, epis_idx: int) -> np.ndarray:
        if epis_idx not in self._cache:
            path = cache_path(self.task, self.model, self.scale, self.camera, epis_idx)
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} missing; run scripts/encode_dino.py --task {self.task}"
                )
            self._cache[epis_idx] = np.load(path, mmap_mode="r")
        return self._cache[epis_idx]
