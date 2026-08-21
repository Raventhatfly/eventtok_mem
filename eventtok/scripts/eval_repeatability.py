"""Measure code stability and repetition recoverability for a checkpoint.

    python -m eventtok.scripts.eval_repeatability --ckpt ... --episodes 30
"""

from __future__ import annotations

import argparse
import json

import torch

from .. import paths
from ..data.index import RoboMMEIndex
from ..data.meta import TaskMeta
from ..data.robomme import TransitionDataset
from ..eval.repeatability import report
from .tokenize_episodes import load_model, stream_for_episode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="SwingXtimes")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    paths.check_root()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = load_model(args.ckpt, device)

    index = RoboMMEIndex()
    eps = index.by_task(args.task)[: args.episodes]
    meta = TaskMeta(args.task)
    ds = TransitionDataset(
        args.task, k=cfg["k"], scale=cfg["scale"], episodes=eps, index=index
    )

    streams = {}
    for ep in eps:
        codes, _, _ = stream_for_episode(model, ds, ep.epis_idx, device)
        streams[ep.epis_idx] = codes

    res = report(streams, {ep.epis_idx: ep for ep in eps}, meta)
    live = len({c for s in streams.values() for c in s})
    print(
        f"levels={tuple(cfg['levels'])} |C|={model.codebook_size} live={live} "
        f"({live/model.codebook_size:.1%})"
    )
    print(
        f"  within-event change rate {res['within_event_change_rate']:.1%} "
        f"(mean code run {res['mean_code_run']:.1f} transitions)"
    )
    print(
        f"  n-gram == N: exact {res['ngram_exact']}/{res['episodes']} "
        f"({res['exact_frac']:.0%})  over {res['ngram_over']}  under {res['ngram_under']}"
    )
    out = paths.CACHE_ROOT / "eval" / f"repeatability_{'x'.join(map(str, cfg['levels']))}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"levels": cfg["levels"], "live_codes": live, **res}, fh, indent=2)
    print("  wrote", out)


if __name__ == "__main__":
    main()
