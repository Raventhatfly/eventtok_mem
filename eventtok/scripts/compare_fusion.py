"""Do SigLIP and DINOv2 carry complementary information? A fair test.

    python -m eventtok.scripts.compare_fusion --task ButtonUnmask --ks 16 32 64 128

``compare_modalities.py`` reported that concatenating the two encoders usually lost
to the better one alone. That comparison was **not fair to the concat**, in two ways
that both inflate the single-encoder rows:

1. **Variance.** Every block is normalised to total variance 1 and then concatenated,
   so ``action + siglip`` weights vision against action 1:1 while
   ``action + siglip + dinov2l`` weights it 2:1 and the four-source row weights it
   4:1. Adding an encoder silently reweighted the distance, so "does a second encoder
   help" was entangled with "does more vision weight help".
2. **Dimensionality.** PCA-64 per source gives the pair 128 components against 64 for
   a single encoder. Not dimension-matched either.

This script fixes both. Vision is normalised **as a group**, so the vision:action
ratio is 1:1 no matter how many sources go in, and three fusion modes are compared at
explicit component budgets:

    joint   concatenate the raw features, then one PCA to n components.
            Same dims and same variance as a single encoder -- the strict test of
            whether the union spans directions neither source has alone.
    split   PCA each source to n/len(sources), concatenate. Same total dims, but each
            source is capped at half the budget.
    stack   PCA each source to n, concatenate (2n dims). More capacity, so a win here
            is not evidence of complementarity by itself -- reported because it is
            what the earlier script did, and the difference is the point.

Read `joint` against the best single encoder at the same k. That is the number that
answers the question.
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

MODES = ("joint", "split", "stack")


def build_vision(
    raw: dict[str, np.ndarray],
    keys: list[str],
    train_mask: np.ndarray,
    n_components: int,
    mode: str,
    seed: int,
) -> np.ndarray:
    """A vision block from one or more sources, normalised as a group to variance 1.

    The group normalisation is the fix: it keeps vision's weight against the action
    block constant however many encoders or cameras are folded in.
    """
    if len(keys) == 1:
        block = Block(keys[0], n_components).fit(raw[keys[0]][train_mask], seed=seed)
        return block.transform(raw[keys[0]])

    if mode == "joint":
        stacked = np.concatenate([raw[k] for k in keys], axis=1)
        block = Block("+".join(keys), n_components).fit(stacked[train_mask], seed=seed)
        out = block.transform(stacked)
    elif mode == "split":
        per = max(n_components // len(keys), 1)
        parts = []
        for k in keys:
            b = Block(k, per).fit(raw[k][train_mask], seed=seed)
            parts.append(b.transform(raw[k]))
        out = np.concatenate(parts, axis=1)
    elif mode == "stack":
        parts = []
        for k in keys:
            b = Block(k, n_components).fit(raw[k][train_mask], seed=seed)
            parts.append(b.transform(raw[k]))
        out = np.concatenate(parts, axis=1)
    else:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    # Renormalise the group: split and stack concatenate blocks that were each scaled
    # to variance 1, which is exactly the reweighting this script exists to remove.
    total = float((out[train_mask].astype(np.float64) ** 2).sum(1).mean())
    return out / float(np.sqrt(max(total, 1e-12)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ButtonUnmask")
    ap.add_argument("--sources", nargs="+", default=["siglip", "dinov2l"])
    ap.add_argument("--ks", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--vision-form", default="both")
    ap.add_argument("--scale", default="2x2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths.check_root()
    index = RoboMMEIndex()
    eps = index.by_task(args.task)
    meta = TaskMeta(args.task)

    order = np.random.default_rng(args.seed).permutation(len(eps))
    cut = len(eps) // 2
    train_eps = {eps[i].epis_idx for i in order[:cut]}

    rows, epis_of_row, frame_of_row = [], [], []
    for ep in eps:
        lo, hi = meta.rows(ep.epis_idx)
        rows.extend(range(lo, hi))
        epis_of_row.extend([ep.epis_idx] * (hi - lo))
        frame_of_row.extend(range(hi - lo))
    rows = np.asarray(rows)
    epis_of_row = np.asarray(epis_of_row)
    frame_of_row = np.asarray(frame_of_row)
    train_mask = np.array([e in train_eps for e in epis_of_row])

    raw: dict[str, np.ndarray] = {}
    for key in args.sources:
        enc, cam = (key[: -len("_wrist")], "wrist_image") if key.endswith("_wrist") else (key, "image")
        if enc == "siglip" and cam == "image":
            getter = repack.EpisodeFeatures(args.task, args.scale)
        else:
            getter = dino.EpisodeDinoFeatures(args.task, enc, args.scale, cam)
        raw[key] = vision_matrix(
            getter, epis_of_row, frame_of_row, args.horizon, args.vision_form
        )
    action_raw = action_matrix(meta, rows)
    action = Block("action", None).fit(action_raw[train_mask], seed=args.seed).transform(action_raw)

    print(
        f"{args.task}: {len(eps)} episodes, {len(rows)} transitions, "
        f"sources={args.sources} pca={args.pca} "
        f"(vision normalised as a group, so vision:action is 1:1 in every row)",
        flush=True,
    )

    def score(X: np.ndarray, k: int) -> tuple[float, float]:
        codes, _ = kmeans_fit_predict(X[train_mask], X, k, args.seed)
        c_all, l_all, t_all = [], [], []
        for ep in eps:
            lo, hi = meta.rows(ep.epis_idx)
            c, l = label_mutual_information(codes[lo:hi].tolist(), ep, meta)
            c_all.extend(c)
            l_all.extend(l)
            t_all.extend([ep.epis_idx in train_eps] * len(c))
        return label_accuracy(c_all, l_all, t_all)

    variants: list[tuple[str, list[str], str]] = [([s], "single") for s in args.sources]
    variants = [(f"{s} alone", [s], "single") for s in args.sources]
    for mode in MODES:
        variants.append((f"{'+'.join(args.sources)} [{mode}]", list(args.sources), mode))

    results = []
    for k in args.ks:
        print(f"\n  --- k={k} ---", flush=True)
        acc_action, majority = score(action, k)
        print(f"    {'action only':34s} dims {action.shape[1]:4d}  acc {acc_action:6.1%}  (maj {majority:.1%})")
        per_k = {"k": k, "action_only": acc_action, "majority": majority, "rows": []}
        for title, keys, mode in variants:
            V = build_vision(raw, keys, train_mask, args.pca, mode, args.seed)
            acc_v, _ = score(V, k)
            X = np.concatenate([action, V], axis=1)
            acc_av, _ = score(X, k)
            per_k["rows"].append(
                {"condition": title, "mode": mode, "sources": keys,
                 "vision_dims": int(V.shape[1]),
                 "vision_only": acc_v, "action_plus": acc_av}
            )
            print(
                f"    {title:34s} dims {V.shape[1]:4d}  "
                f"vision-only {acc_v:6.1%}   action+ {acc_av:6.1%}",
                flush=True,
            )

        singles = [r for r in per_k["rows"] if r["mode"] == "single"]
        joint = next(r for r in per_k["rows"] if r["mode"] == "joint")
        best_single_v = max(singles, key=lambda r: r["vision_only"])
        best_single_a = max(singles, key=lambda r: r["action_plus"])
        per_k["fusion_gain_vision"] = joint["vision_only"] - best_single_v["vision_only"]
        per_k["fusion_gain_action_plus"] = joint["action_plus"] - best_single_a["action_plus"]
        print(
            f"    -> joint fusion vs best single, matched dims: "
            f"vision-only {per_k['fusion_gain_vision']:+.1%}  "
            f"action+ {per_k['fusion_gain_action_plus']:+.1%}"
        )
        results.append(per_k)

    gv = [r["fusion_gain_vision"] for r in results]
    ga = [r["fusion_gain_action_plus"] for r in results]
    print(f"\n  joint fusion gain over the best single encoder, matched dims and variance:")
    print(f"    vision-only  " + "  ".join(f"k={r['k']}: {r['fusion_gain_vision']:+.1%}" for r in results))
    print(f"    action+      " + "  ".join(f"k={r['k']}: {r['fusion_gain_action_plus']:+.1%}" for r in results))
    verdict = (
        "complementary -- fusion beats either encoder alone at every k"
        if min(gv) > 0 and min(ga) > 0
        else "not complementary -- fusion loses to a single encoder at some k"
        if max(gv) <= 0 and max(ga) <= 0
        else "MIXED -- fusion helps at some k and not others; report the whole row"
    )
    print(f"    verdict: {verdict}")

    out = args.out or str(paths.CACHE_ROOT / "eval" / f"fusion_{args.task}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"task": args.task, "sources": args.sources, "pca": args.pca,
                   "results": results, "verdict": verdict}, fh, indent=2)
    print("\n  wrote", out)


if __name__ == "__main__":
    main()
