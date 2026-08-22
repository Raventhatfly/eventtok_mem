"""How big does a difference have to be before it means anything?

    python -m eventtok.scripts.noise_floor --task ButtonUnmask --k 32 --seeds 10

Every comparison in this project was run at seed 0 -- one episode split, one k-means
initialisation -- and differences of 1-5 points were read as results. Several of them
later reversed. This measures the error bar those numbers should have carried.

Two quantities, and the second is the one that matters:

  * spread of a single condition across seeds. Interesting but pessimistic, since a
    comparison run at a shared seed cancels the part of the noise that comes from
    which episodes landed in the held-out half.
  * spread of the *paired difference* between two conditions across seeds. This is
    the correct error bar for "encoder A beats encoder B by x", because that is how
    the comparison was actually made.

Report a difference only when it is large against the paired sd. The convention used
here is 2 sd, which for these split sizes is the difference between a finding and a
coin flip dressed as one.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .. import paths
from ..data import dino, repack
from ..data.index import RoboMMEIndex
from ..data.meta import TaskMeta
from ..eval.repeatability import label_accuracy, label_mutual_information
from .compare_modalities import Block, action_matrix, kmeans_fit_predict, vision_matrix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ButtonUnmask")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--scale", default="2x2")
    ap.add_argument(
        "--sources", nargs="+", default=["siglip", "dinov2l", "siglip_wrist"],
        help="vision sources to score; pairwise differences are reported for all",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths.check_root()
    index = RoboMMEIndex()
    eps = index.by_task(args.task)
    meta = TaskMeta(args.task)

    rows, epis_of_row, frame_of_row = [], [], []
    for ep in eps:
        lo, hi = meta.rows(ep.epis_idx)
        rows.extend(range(lo, hi))
        epis_of_row.extend([ep.epis_idx] * (hi - lo))
        frame_of_row.extend(range(hi - lo))
    rows = np.asarray(rows)
    epis_of_row = np.asarray(epis_of_row)
    frame_of_row = np.asarray(frame_of_row)

    raw = {"action": action_matrix(meta, rows)}
    for key in args.sources:
        enc, cam = (
            (key[: -len("_wrist")], "wrist_image") if key.endswith("_wrist") else (key, "image")
        )
        getter = (
            repack.EpisodeFeatures(args.task, args.scale)
            if (enc == "siglip" and cam == "image")
            else dino.EpisodeDinoFeatures(args.task, enc, args.scale, cam)
        )
        raw[key] = vision_matrix(getter, epis_of_row, frame_of_row, args.horizon, "both")

    print(
        f"{args.task}: k={args.k}, {args.seeds} seeds, sources={args.sources}\n"
        f"  each seed redraws the 50/50 episode split AND the k-means init, "
        f"which is what a rerun of any earlier comparison would have done",
        flush=True,
    )

    per_seed: dict[str, list[float]] = {s: [] for s in args.sources}
    for seed in range(args.seeds):
        order = np.random.default_rng(seed).permutation(len(eps))
        train_eps = {eps[i].epis_idx for i in order[: len(eps) // 2]}
        train_mask = np.array([e in train_eps for e in epis_of_row])

        for src in args.sources:
            block = Block(src, args.pca).fit(raw[src][train_mask], seed=seed)
            X = block.transform(raw[src])
            codes, _ = kmeans_fit_predict(X[train_mask], X, args.k, seed)
            c_all, l_all, t_all = [], [], []
            for ep in eps:
                lo, hi = meta.rows(ep.epis_idx)
                c, l = label_mutual_information(codes[lo:hi].tolist(), ep, meta)
                c_all.extend(c)
                l_all.extend(l)
                t_all.extend([ep.epis_idx in train_eps] * len(c))
            acc, _ = label_accuracy(c_all, l_all, t_all)
            per_seed[src].append(acc)
        print(
            f"  seed {seed}: "
            + "  ".join(f"{s} {per_seed[s][-1]:.1%}" for s in args.sources),
            flush=True,
        )

    print("\n  single-condition spread across seeds:")
    for s in args.sources:
        v = np.array(per_seed[s])
        print(f"    {s:16s} mean {v.mean():.1%}  sd {v.std(ddof=1):.1%}  "
              f"range {v.min():.1%}-{v.max():.1%}")

    print("\n  PAIRED difference across seeds -- the error bar earlier claims needed:")
    pairs = []
    for i, a in enumerate(args.sources):
        for b in args.sources[i + 1:]:
            d = np.array(per_seed[a]) - np.array(per_seed[b])
            sd = float(d.std(ddof=1))
            mean = float(d.mean())
            flips = int(((d > 0).sum(), (d < 0).sum())[0] not in (0, len(d)))
            pairs.append({"a": a, "b": b, "mean": mean, "sd": sd,
                          "min": float(d.min()), "max": float(d.max()),
                          "sign_flips": bool(flips)})
            verdict = (
                "sign is STABLE" if (d > 0).all() or (d < 0).all()
                else "SIGN FLIPS across seeds -- not a result"
            )
            print(
                f"    {a:16s} - {b:16s} mean {mean:+.1%}  sd {sd:.1%}  "
                f"2sd {2*sd:.1%}  range {d.min():+.1%}..{d.max():+.1%}   [{verdict}]"
            )

    worst = max(pairs, key=lambda p: p["sd"])
    print(
        f"\n  a difference smaller than ~{2*worst['sd']:.1%} at this k and split size "
        f"should not be reported as a finding"
    )

    out = args.out or str(paths.CACHE_ROOT / "eval" / f"noise_{args.task}_k{args.k}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"task": args.task, "k": args.k, "seeds": args.seeds,
                   "per_seed": per_seed, "pairs": pairs}, fh, indent=2)
    print("  wrote", out)


if __name__ == "__main__":
    main()
