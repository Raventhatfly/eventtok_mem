"""One pass over a task's pkls, caching actions + state + subgoal labels.

    python -m eventtok.scripts.build_meta --task SwingXtimes
"""

from __future__ import annotations

import argparse

from .. import paths
from ..data import meta
from ..data.index import RoboMMEIndex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    paths.check_root()
    index = RoboMMEIndex()
    episodes = index.by_task(args.task)
    n = sum(ep.n_exec for ep in episodes)
    print(f"{args.task}: {len(episodes)} episodes, {n} execution frames", flush=True)
    print(f"reading ~{n * 0.4 / 1000:.1f} GB of pkls -> {meta.cache_path(args.task)}", flush=True)

    path = meta.build(args.task, index=index, overwrite=args.overwrite, workers=args.workers)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
