"""Encode a task's frames with DINOv2 into the per-episode feature cache.

    python -m eventtok.scripts.encode_dino --task SwingXtimes --model dinov2l

The point is to answer one question -- does a spatially-detailed encoder beat the
language-aligned SigLIP features for event identity -- so this writes into the same
layout the SigLIP cache uses and changes nothing else. See ``eventtok.data.dino``
for the list of things kept identical on purpose.

Preemption: the per-episode file is the checkpoint. On SIGUSR1 the current episode
is finished, then the process exits ``REQUEUE_EXIT_CODE`` so the batch script can
requeue; the next run skips every episode already on disk. Nothing to reload, no
resume state to get wrong.

Cost is dominated by reading pkls (~400 KB x 2 cameras x n_frames, on netscratch),
not by the GPU. Raise ``--readers`` before reaching for a bigger batch.
"""

from __future__ import annotations

import argparse
import sys
import time

from .. import paths
from ..data import dino
from ..data.index import RoboMMEIndex
from ..train.checkpoint import REQUEUE_EXIT_CODE, PreemptionHandler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="SwingXtimes")
    ap.add_argument("--model", default="dinov2l", choices=sorted(dino.MODELS))
    ap.add_argument("--scale", default="2x2", choices=["2x2", "4x4", "8x8"])
    ap.add_argument(
        "--cameras",
        nargs="+",
        default=list(dino.CAMERAS),
        choices=list(dino.CAMERAS),
        help="wrist_image is a separate variable from the encoder swap; it is "
        "extracted here only because the pkl read is already paid for.",
    )
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--readers", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    paths.check_root()
    index = RoboMMEIndex()
    eps = index.by_task(args.task)
    if args.episodes:
        eps = eps[: args.episodes]

    pending = [
        ep
        for ep in eps
        if args.overwrite
        or any(
            not dino.cache_path(ep.task, args.model, args.scale, c, ep.epis_idx).is_file()
            for c in args.cameras
        )
    ]
    print(
        f"{args.task}: {len(eps)} episodes, {len(pending)} to encode "
        f"({len(eps) - len(pending)} already cached), "
        f"{sum(ep.n_frames for ep in pending)} frames x {len(args.cameras)} cameras",
        flush=True,
    )
    if not pending:
        print("nothing to do")
        return 0

    preempt = PreemptionHandler().install()
    encoder = dino.Dinov2Encoder(args.model, batch=args.batch)
    print(
        f"loaded {args.model} (width {encoder.width}) on {encoder.device}; "
        f"out dir {dino.cache_dir(args.task, args.model, args.scale, 'image')}",
        flush=True,
    )

    done = 0
    for i, ep in enumerate(pending):
        t0 = time.time()
        dino.encode_episode(
            ep,
            encoder,
            scale=args.scale,
            cameras=args.cameras,
            readers=args.readers,
            overwrite=args.overwrite,
        )
        done += 1
        dt = time.time() - t0
        print(
            f"[{i + 1:3d}/{len(pending)}] ep{ep.epis_idx} T={ep.n_frames:4d} "
            f"{dt:5.1f}s ({ep.n_frames / max(dt, 1e-6):5.0f} frame/s)",
            flush=True,
        )
        # Checked between episodes: a file is either complete or absent, so
        # stopping here can never leave a partial episode in the cache.
        if preempt.should_stop:
            print(
                f"[preempt] stopping after {done}/{len(pending)} episodes; "
                f"exit {REQUEUE_EXIT_CODE} to requeue",
                flush=True,
            )
            return REQUEUE_EXIT_CODE

    print(f"encoded {done} episodes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
