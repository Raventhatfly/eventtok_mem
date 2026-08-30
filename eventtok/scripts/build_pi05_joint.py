"""Fit one event vocabulary across all RoboMME tasks and cache every frame's log.

    python -m eventtok.scripts.build_pi05_joint --tag joint

Separate from ``build_pi05_events`` because it cannot be an array job: the whole point
is that one fit produces every task's cache, so the tasks are not independent.
"""

from __future__ import annotations

import argparse

import numpy as np

from .. import paths
from ..pi05 import joint
from ..pi05 import tokens as evt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="joint")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--min-span", type=int, default=3)
    ap.add_argument("--min-frequency", type=int, default=10)
    ap.add_argument("--max-token-length", type=int, default=4)
    ap.add_argument("--max-log", type=int, default=64)
    ap.add_argument("--scale", default="2x2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit-episodes", type=int, default=12)
    ap.add_argument("--fit-rows", type=int, default=60_000)
    ap.add_argument("--vision-weight", type=float, default=1.0)
    args = ap.parse_args()

    paths.check_root()
    out_dir = evt.cache_path("x", args.tag).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    def write(task: str, blob: dict) -> None:
        out = evt.cache_path(task, args.tag)
        tmp = out.with_name(out.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **blob)
        tmp.rename(out)
        print(f"    wrote {out}", flush=True)

    joint.build_joint(
        args.tasks, k=args.k, chunk=args.chunk, min_span=args.min_span,
        min_frequency=args.min_frequency, max_token_length=args.max_token_length,
        max_log=args.max_log, scale=args.scale, seed=args.seed,
        fit_episodes=args.fit_episodes, fit_rows=args.fit_rows,
        vision_weight=args.vision_weight, on_task=write,
    )


if __name__ == "__main__":
    main()
