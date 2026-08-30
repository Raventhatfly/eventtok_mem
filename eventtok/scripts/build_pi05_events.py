"""Precompute the per-frame causal event log for a RoboMME task.

    python -m eventtok.scripts.build_pi05_events --task SwingXtimes

One cache per task, consumed by the pi0.5 input transform. Deterministic given the
tokenizer settings, so it is built once rather than recomputed in the data loader.
"""

from __future__ import annotations

import argparse

import numpy as np

from .. import paths
from ..pi05 import tokens as evt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--tokens", default="action+vision",
                    choices=["action", "action+vision"])
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--min-span", type=int, default=3)
    ap.add_argument("--min-frequency", type=int, default=10)
    ap.add_argument("--max-token-length", type=int, default=4)
    ap.add_argument("--max-log", type=int, default=64)
    ap.add_argument("--scale", default="2x2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths.check_root()
    out = evt.cache_path(args.task, args.tag)
    out.parent.mkdir(parents=True, exist_ok=True)

    blob = evt.build_task(
        args.task, tokens=args.tokens, k=args.k, chunk=args.chunk,
        min_span=args.min_span, min_frequency=args.min_frequency,
        max_token_length=args.max_token_length, max_log=args.max_log,
        scale=args.scale, seed=args.seed,
    )
    # np.savez appends .npz to a path lacking it, which would misname a temp file;
    # write through a handle so the atomic rename lands where intended.
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **blob)
    tmp.rename(out)

    lens = blob["lengths"].astype(int)
    ovf = blob["overflow"].sum(axis=1)
    print(
        f"{args.task}: {len(lens)} frames  "
        f"log {lens.mean():.1f} tokens mean / {lens.max()} max  "
        f"empty {100 * (lens == 0).mean():.0f}% of frames  "
        f"overflowed on {100 * (ovf > 0).mean():.1f}% of frames",
        flush=True,
    )
    print(f"  vocab {blob['meta'] is not None and ''}symbols, wrote {out}", flush=True)


if __name__ == "__main__":
    main()
