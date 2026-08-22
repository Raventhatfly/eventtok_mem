"""SigLIP features for the wrist camera, to remove a confound from the encoder table.

    /n/lab_storage/ydu_lab/wfy/.conda/envs/robomme-openpi/bin/python \
        -m eventtok.scripts.encode_siglip_wrist --task ButtonUnmask

Why this is needed. The cached RoboMME features were built with ``num_views=1`` over
the third-person ``image`` only, so there are no SigLIP wrist features anywhere. That
made every wrist row in the encoder comparison change *two* variables at once --
encoder and camera -- and a two-variable row cannot support a claim about either.
The wrist camera turned out to matter more than the encoder choice on ButtonUnmask
(+11.4 points over the base camera at fixed encoder), which is exactly why the
matched row has to exist rather than be waved at.

This runs pi0.5's own SigLIP tokenizer, the same weights that produced the cached
base-camera features (``siglip_params.pkl``, So400m/14, num_classes=2048), so the
only difference from those is the camera. It needs the ``robomme-openpi`` env: JAX
0.5.3 with the CUDA12 plugin. The rest of the project runs under ``robomme``, which
has no JAX -- hence a separate entry point rather than a flag on encode_dino.

Preprocessing follows ``mem_buffer.add_buffer`` exactly: ``/255 * 2 - 1`` to [-1, 1],
``resize_with_pad`` to 224 (a no-op crop-wise on square 256x256 input), then
``pool_tokens_to_size``. Output lands in the same per-episode layout as the DINOv2
cache under the model name ``siglip``, so ``compare_modalities`` reads it with no
special case.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

DEFAULT_OPENPI_DATA = "/n/netscratch/ydu_lab/Lab/wfy/dataset/robomme/openpi_data"
POLICY_SRC = "/n/home04/wfy/repos/robomme_policy_learning/src"

_N_TOKENS = {"2x2": 4, "4x4": 16, "8x8": 64}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ButtonUnmask")
    ap.add_argument("--camera", default="wrist_image", choices=["wrist_image", "image"])
    ap.add_argument("--scale", default="2x2", choices=sorted(_N_TOKENS))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--readers", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--openpi-data", default=DEFAULT_OPENPI_DATA)
    ap.add_argument("--policy-src", default=POLICY_SRC)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("OPENPI_DATA_HOME", args.openpi_data)
    if args.policy_src not in sys.path:
        sys.path.insert(0, args.policy_src)

    from .. import paths
    from ..data import dino
    from ..data.index import RoboMMEIndex
    from ..train.checkpoint import REQUEUE_EXIT_CODE, PreemptionHandler

    paths.check_root()
    index = RoboMMEIndex()
    eps = index.by_task(args.task)
    if args.episodes:
        eps = eps[: args.episodes]

    def out_path(ep):
        return dino.cache_path(ep.task, "siglip", args.scale, args.camera, ep.epis_idx)

    pending = [ep for ep in eps if args.overwrite or not out_path(ep).is_file()]
    print(
        f"{args.task}: {len(eps)} episodes, {len(pending)} to encode, "
        f"{sum(ep.n_frames for ep in pending)} frames, camera={args.camera}",
        flush=True,
    )
    if not pending:
        print("nothing to do")
        return 0

    preempt = PreemptionHandler().install()

    import jax
    import jax.numpy as jnp
    from mme_vla_suite.shared.data_utils import pool_tokens_to_size
    from mme_vla_suite.shared.siglip_tokenizer import SigLipTokenizer
    from openpi.shared import image_tools

    print(f"jax devices: {jax.devices()}", flush=True)
    tokenizer = SigLipTokenizer(inference_batch_size=args.batch)
    target = _N_TOKENS[args.scale]

    # jit, and feed a *fixed* batch shape. Both matter: the un-jitted nnx call runs
    # eagerly at ~6 frame/s against ~190 for the torch DINOv2 path, and jit
    # retraces on every new input shape, so a ragged final batch would recompile
    # once per episode and give most of the speedup back. Short batches are padded
    # and trimmed instead.
    encode_batch = jax.jit(tokenizer.__call__)
    print(f"loaded pi0.5 SigLIP; out dir {out_path(pending[0]).parent}", flush=True)

    import time

    for i, ep in enumerate(pending):
        t0 = time.time()
        images = dino.read_cameras(ep, (args.camera,), workers=args.readers)[args.camera]
        if len(images) != ep.n_frames:
            raise ValueError(
                f"episode {ep.epis_idx}: read {len(images)} frames, index says "
                f"{ep.n_frames}"
            )

        chunks = []
        for lo in range(0, len(images), args.batch):
            batch = images[lo : lo + args.batch]
            n = len(batch)
            if n < args.batch:
                # Pad to the fixed shape so jit does not retrace, then trim below.
                pad = np.repeat(batch[-1:], args.batch - n, axis=0)
                batch = np.concatenate([batch, pad], axis=0)
            # mem_buffer.add_buffer, verbatim: to [-1, 1] first, then resize_with_pad.
            x = jnp.array(batch.astype(np.float32) / 255.0 * 2.0 - 1.0)
            x = image_tools.resize_with_pad(x, 224, 224)
            out = encode_batch(x[:, None])                   # (b, 1, p, 2048)
            pooled = pool_tokens_to_size(out, target)        # (b, 1, target, 2048)
            chunks.append(np.asarray(jax.device_get(pooled))[:n, 0])
        feats = np.concatenate(chunks, axis=0).astype(np.float16)

        expected = (ep.n_frames, target, 2048)
        if feats.shape != expected:
            raise ValueError(f"episode {ep.epis_idx}: {feats.shape} != {expected}")

        out = out_path(ep)
        out.parent.mkdir(parents=True, exist_ok=True)
        # np.save() appends ".npy" to a path that lacks it, which would misname the
        # temp file and break the atomic rename.
        tmp = out.with_name(out.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, feats)
        tmp.rename(out)

        dt = time.time() - t0
        print(
            f"[{i + 1:3d}/{len(pending)}] ep{ep.epis_idx} T={ep.n_frames:4d} "
            f"{dt:5.1f}s ({ep.n_frames / max(dt, 1e-6):5.0f} frame/s)",
            flush=True,
        )
        if preempt.should_stop:
            print(f"[preempt] stopping after {i + 1}/{len(pending)}", flush=True)
            return REQUEUE_EXIT_CODE

    print(f"encoded {len(pending)} episodes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
