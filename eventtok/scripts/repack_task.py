"""Repack one task's per-frame features into per-episode arrays.

    python -m eventtok.scripts.repack_task --task SwingXtimes
"""

from __future__ import annotations

import argparse

from ..data import repack
from ..data.index import RoboMMEIndex
from .. import paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--scale", default="2x2", choices=repack.SCALES)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="first N episodes only")
    args = ap.parse_args()

    paths.check_root()
    index = RoboMMEIndex()
    episodes = index.by_task(args.task)
    if args.limit is not None:
        episodes = episodes[: args.limit]

    total_frames = sum(ep.n_frames for ep in episodes)
    n_tok = repack._N_TOKENS[args.scale]
    mb = total_frames * n_tok * 2048 * 2 / 1e6
    print(
        f"{args.task}: {len(episodes)} episodes, {total_frames} frames, "
        f"scale {args.scale} -> ~{mb:.0f} MB",
        flush=True,
    )
    print(f"cache: {repack.cache_dir(args.task, args.scale)}", flush=True)

    for i, ep in enumerate(episodes):
        path = repack.repack_episode(
            ep, scale=args.scale, workers=args.workers, overwrite=args.overwrite
        )
        print(f"[{i + 1:3d}/{len(episodes)}] ep{ep.epis_idx} T={ep.n_frames:4d} -> {path.name}", flush=True)


if __name__ == "__main__":
    main()
