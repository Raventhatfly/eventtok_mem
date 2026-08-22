"""Which inputs make an event code identify the event? One table, one variable at a time.

    python -m eventtok.scripts.compare_modalities --task SwingXtimes \
        --encoders siglip dinov2l --k 32

This replaces the ad-hoc runs behind the earlier modality table, which were never
committed and so could not be re-checked when the metric changed. Every row here
comes from the same k-means, the same split, the same metric.

Three things it fixes relative to those runs.

**1. The features are centred and energy-normalised.** Measured on SwingXtimes, the
shared mean accounts for **99.8%** of SigLIP's feature energy and 94.9% of
DINOv2's, so Euclidean k-means on raw features spends nearly all its distance
budget on a constant. Each block is centred with train statistics and scaled so
its total variance is 1, which also makes "action" and "vision" contribute equally
to the distance instead of letting the wider block win by dimension count.

**2. Vision is reduced to a fixed number of PCA components (default 64) for every
encoder.** SigLIP here is 4x2048 and DINOv2-large 4x1024, so comparing them at
native width would confound "which encoder" with "how many dimensions". Same
component count for both leaves the encoder as the only difference.

**3. The split is by episode, and k-means is fit on train episodes only.** Adjacent
frames are nearly identical, so a frame-level split leaks the answer across the
boundary and every number comes out high.

Report the majority baseline on the same line as the accuracy, always. The majority
class is ~50% on ButtonUnmask and ~30% on SwingXtimes; an accuracy without it is
unreadable, which is a mistake this project has already made four times.

What this script has measured so far, with camera and encoder varied one at a time
(vision-only rows, held-out episodes, swept over k because the first version of this
result was read off k=32 and was wrong):

    camera effect, encoder fixed (siglip base -> siglip wrist)
        ButtonUnmask   +6.7  +11.9   +8.7   +7.2      (k = 16, 32, 64, 128)
        SwingXtimes    +3.6   +0.8   +0.9   +1.7

    encoder effect, camera fixed (siglip -> dinov2l)
        ButtonUnmask   +2.5   +6.0   +2.7   +2.4
        SwingXtimes    -1.2   -0.8   -2.1   -0.4

So the encoder is close to a wash -- DINOv2 is consistently *worse* than SigLIP as a
standalone source on SwingXtimes -- while the camera is worth several times as much
and is positive on both tasks. The available gain was a camera the pipeline was
discarding: RoboMME's shipped features use num_views=1 over the third-person image,
so wrist_image sits unused in every pkl. Once the wrist camera is in, the encoder
matters even less (dinov2l_wrist and siglip_wrist within 0.5 points at k >= 32 on
ButtonUnmask). action + siglip_wrist reaches 98.6% at k=128 against 51.8% majority.

Two cautions carried by the same grid. The best single condition changes with k on
both tasks, so no one row is the result -- though every winner at every k does include
the wrist camera. And label accuracy rises monotonically with k, so it cannot be used
alone to pick a codebook size; see scripts/sweep_k_tradeoff.py, where counting stays
at 92-100% across the whole range while the stream fragments from 26 to 76 run
symbols per episode.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .. import paths
from ..data import dino, repack
from ..data.index import RoboMMEIndex
from ..data.meta import TaskMeta
from ..eval.repeatability import label_accuracy, label_mutual_information

VISION_FORMS = ("state", "residual", "both")


class Block:
    """One feature block: centre, optionally project, then normalise its energy.

    Statistics come from the train rows only. The energy normalisation is what
    makes blocks combinable -- after it, each block contributes total variance 1
    to the Euclidean distance regardless of its width.
    """

    def __init__(self, name: str, n_components: int | None = None) -> None:
        self.name = name
        self.n_components = n_components
        self.mu: np.ndarray | None = None
        self.basis: np.ndarray | None = None
        self.scale: float = 1.0

    def fit(self, X: np.ndarray, seed: int = 0) -> "Block":
        X = np.asarray(X, dtype=np.float32)
        self.mu = X.mean(0)
        Xc = X - self.mu
        if self.n_components and self.n_components < Xc.shape[1]:
            # Randomised range finder: exact SVD on (n x 8192) is needlessly slow
            # and the leading components are all that is used.
            rng = np.random.default_rng(seed)
            probe = rng.standard_normal((Xc.shape[1], self.n_components + 16), dtype=np.float32)
            Y = Xc @ probe
            Q, _ = np.linalg.qr(Y)
            _, _, Vt = np.linalg.svd(Q.T @ Xc, full_matrices=False)
            self.basis = Vt[: self.n_components].T.copy()
            Xc = Xc @ self.basis
        total = float((Xc.astype(np.float64) ** 2).sum(1).mean())
        self.scale = float(np.sqrt(max(total, 1e-12)))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xc = np.asarray(X, dtype=np.float32) - self.mu
        if self.basis is not None:
            Xc = Xc @ self.basis
        return Xc / self.scale

    @property
    def width(self) -> int:
        return self.n_components if self.basis is not None else len(self.mu)


def action_matrix(meta: TaskMeta, rows: np.ndarray) -> np.ndarray:
    scale = meta.action_scale
    return np.stack([(meta.delta_actions(int(r)) / scale).ravel() for r in rows])


def vision_matrix(
    getter, epis_idx_of_row: np.ndarray, frame_of_row: np.ndarray, k: int, form: str
) -> np.ndarray:
    """``feat_t`` and/or ``feat_{t+k} - feat_t``, flattened, per row.

    The residual is included because the state alone is nearly constant over k
    frames -- the same reason the prediction head collapsed when it was asked to
    reproduce ``feat_next`` directly rather than the change.
    """
    out = []
    for epis_idx, t in zip(epis_idx_of_row, frame_of_row):
        arr = getter[int(epis_idx)]
        t = int(t)
        nxt = min(t + k, len(arr) - 1)
        a = np.asarray(arr[t], dtype=np.float32).ravel()
        b = np.asarray(arr[nxt], dtype=np.float32).ravel()
        if form == "state":
            out.append(a)
        elif form == "residual":
            out.append(b - a)
        else:
            out.append(np.concatenate([a, b - a]))
    return np.stack(out)


def kmeans_fit_predict(train: np.ndarray, allX: np.ndarray, k: int, seed: int):
    from scipy.cluster.vq import kmeans2

    np.random.seed(seed)
    centroids, _ = kmeans2(train, k, minit="++", seed=seed)
    d = ((allX[:, None, :] - centroids[None]) ** 2).sum(-1)
    return d.argmin(1), centroids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="SwingXtimes")
    ap.add_argument("--encoders", nargs="+", default=["siglip", "dinov2l"])
    ap.add_argument(
        "--cameras",
        nargs="+",
        default=["image"],
        help="A wrist row and a base row differ in camera as well as encoder, so "
        "do not read across them as an encoder comparison. siglip+wrist needs "
        "scripts/encode_siglip_wrist.py to have been run first.",
    )
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--vision-form", default="both", choices=VISION_FORMS)
    ap.add_argument("--scale", default="2x2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths.check_root()
    index = RoboMMEIndex()
    eps = index.by_task(args.task)
    if args.episodes:
        eps = eps[: args.episodes]
    meta = TaskMeta(args.task)

    # Split by episode. A frame-level split leaks: consecutive frames differ by
    # almost nothing, so a neighbour of every test frame would be in train.
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
    print(
        f"{args.task}: {len(eps)} episodes, {len(rows)} transitions, "
        f"{train_mask.sum()} train / {(~train_mask).sum()} test, "
        f"k={args.k} horizon={args.horizon} pca={args.pca} form={args.vision_form}",
        flush=True,
    )

    # --- raw blocks -------------------------------------------------------
    raw: dict[str, np.ndarray] = {"action": action_matrix(meta, rows)}
    for enc in args.encoders:
        for cam in args.cameras:
            key = enc if cam == "image" else f"{enc}_wrist"
            if enc == "siglip" and cam == "image":
                # The shipped RoboMME cache, built with num_views=1 over the
                # third-person camera. Everything else comes from a cache this
                # repo wrote, including SigLIP's wrist features -- see
                # scripts/encode_siglip_wrist.py for why that row has to exist.
                getter = repack.EpisodeFeatures(args.task, args.scale)
            else:
                getter = dino.EpisodeDinoFeatures(args.task, enc, args.scale, cam)
            raw[key] = vision_matrix(
                getter, epis_of_row, frame_of_row, args.horizon, args.vision_form
            )
    for name, X in raw.items():
        print(f"  block {name:16s} {X.shape}", flush=True)

    # --- fit each block on train rows -------------------------------------
    blocks = {
        name: Block(name, None if name == "action" else args.pca).fit(
            X[train_mask], seed=args.seed
        )
        for name, X in raw.items()
    }
    feats = {name: blocks[name].transform(raw[name]) for name in raw}

    vision_keys = [n for n in raw if n != "action"]
    conditions: list[tuple[str, list[str]]] = [("action only (= OAT)", ["action"])]
    for v in vision_keys:
        conditions.append((f"{v} only", [v]))
    for v in vision_keys:
        conditions.append((f"action + {v}", ["action", v]))
    if len(vision_keys) > 1:
        conditions.append(("action + " + " + ".join(vision_keys), ["action", *vision_keys]))

    # Labels are fixed across conditions; build them once, per episode, so the
    # frame ordering matches the row ordering exactly.
    ep_by_idx = {ep.epis_idx: ep for ep in eps}
    results = []
    for title, parts in conditions:
        X = np.concatenate([feats[p] for p in parts], axis=1)
        codes, _ = kmeans_fit_predict(X[train_mask], X, args.k, args.seed)

        all_codes, all_labels, all_train = [], [], []
        for ep in eps:
            lo, hi = meta.rows(ep.epis_idx)
            c, l = label_mutual_information(codes[lo:hi].tolist(), ep, meta)
            all_codes.extend(c)
            all_labels.extend(l)
            all_train.extend([ep.epis_idx in train_eps] * len(c))
        acc, base = label_accuracy(all_codes, all_labels, all_train)
        live = len(set(all_codes))
        row = {
            "condition": title,
            "parts": parts,
            "dims": int(X.shape[1]),
            "accuracy": acc,
            "majority": base,
            "gain": acc - base,
            "live_clusters": live,
            "n_labels": len(set(all_labels)),
        }
        results.append(row)
        print(
            f"  {title:34s} dims {X.shape[1]:5d}  acc {acc:6.1%}  "
            f"majority {base:6.1%}  (+{acc - base:5.1%})  live {live}/{args.k}",
            flush=True,
        )

    best_action = next(r for r in results if r["parts"] == ["action"])
    print("\n  gain over action-only (the OAT comparison):")
    for r in results:
        if r["parts"] == ["action"]:
            continue
        print(
            f"    {r['condition']:34s} {r['accuracy'] - best_action['accuracy']:+6.1%}"
        )

    out = args.out or str(
        paths.CACHE_ROOT / "eval" / f"modalities_{args.task}_k{args.k}.json"
    )
    payload = {
        "task": args.task,
        "k": args.k,
        "horizon": args.horizon,
        "pca": args.pca,
        "vision_form": args.vision_form,
        "seed": args.seed,
        "episodes": len(eps),
        "transitions": int(len(rows)),
        "results": results,
    }
    import os

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\n  wrote", out)


if __name__ == "__main__":
    main()
