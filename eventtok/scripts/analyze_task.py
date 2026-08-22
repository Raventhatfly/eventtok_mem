"""Run the whole tokenizer pipeline on one RoboMME task and write a JSON row.

    python -m eventtok.scripts.analyze_task --task InsertPeg

One cell of the 16-task analysis. Everything the project has measured was measured on
two or three tasks -- SwingXtimes, ButtonUnmask, PickXtimes -- and every conclusion
that had to be retracted came from generalising one of them. This runs the same
pipeline, the same split and the same metrics on whatever task it is given, so the
question "where does this work" can be answered from evidence rather than from the
tasks that happened to be prepped first.

Reported per task, all on held-out episodes with k-means and BPE fitted on the other
half:

  structure     episodes, execution frames, annotated events per episode, how many
                distinct observable labels, and the majority-label rate -- the last is
                the baseline every accuracy has to be read against, and it varies from
                ~28% to over 50% across tasks.
  identity      label accuracy from action codes, and from action+vision codes.
  boundaries    F1 of run-symbol boundaries and of BPE token boundaries, plus how far
                the token count sits from the true event count.

Tasks whose events are motions should do well from actions alone; tasks whose events
are defined by something off the trajectory -- a demo video, a hidden object -- should
not, and the point of running all 16 is to find out whether that split is real.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np

from .. import paths
from ..bpe import build_vocab as bpe
from ..data import repack, subgoals as sg
from ..data.index import RoboMMEIndex
from ..data.meta import TaskMeta
from ..eval.bpe_boundaries import report as boundary_report, runs_with_spans
from ..eval.repeatability import label_accuracy, label_mutual_information
from ..eval.token_identity import report as token_report
from .compare_modalities import Block, action_matrix, kmeans_fit_predict, vision_matrix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--scale", default="2x2")
    ap.add_argument("--min-span", type=int, default=3)
    ap.add_argument("--min-frequency", type=int, default=10)
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--tolerance", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths.check_root()
    index = RoboMMEIndex()
    eps = index.by_task(args.task)
    meta = TaskMeta(args.task)

    order = np.random.default_rng(args.seed).permutation(len(eps))
    cut = len(eps) // 2
    train_eps = [eps[i] for i in order[:cut]]
    test_eps = [eps[i] for i in order[cut:]]
    train_set = {ep.epis_idx for ep in train_eps}

    rows, epis_of_row, frame_of_row, exec_of_row = [], [], [], []
    for ep in eps:
        lo, hi = meta.rows(ep.epis_idx)
        rows.extend(range(lo, hi))
        epis_of_row.extend([ep.epis_idx] * (hi - lo))
        frame_of_row.extend(range(hi - lo))
        exec_of_row.extend([ep.exec_start] * (hi - lo))
    rows = np.asarray(rows)
    epis_of_row = np.asarray(epis_of_row)
    frame_of_row = np.asarray(frame_of_row)
    exec_of_row = np.asarray(exec_of_row)
    train_mask = np.array([e in train_set for e in epis_of_row])

    # ---- structure --------------------------------------------------------
    seg_counts, labels_seen = [], Counter()
    for ep in eps:
        segs = sg.segments_from_track(meta.episode_labels(ep.epis_idx), ep.exec_start)
        seg_counts.append(len(segs))
        for s in segs:
            labels_seen[sg.observable_label(s.label)] += 1
    frame_labels = [sg.observable_label(str(x)) for x in meta.labels]
    majority = Counter(frame_labels).most_common(1)[0][1] / max(len(frame_labels), 1)

    row = {
        "task": args.task,
        "k": args.k,
        "episodes": len(eps),
        "exec_frames": int(len(rows)),
        "frames_per_episode": float(len(rows) / len(eps)),
        "events_per_episode": float(np.mean(seg_counts)),
        "distinct_labels": len(labels_seen),
        "majority_label_rate": majority,
        "has_prefix": bool(any(ep.exec_start > 0 for ep in eps)),
    }
    print(
        f"{args.task}: {len(eps)} eps, {len(rows)} exec frames, "
        f"{row['events_per_episode']:.1f} events/ep, {len(labels_seen)} labels, "
        f"majority {majority:.1%}, prefix={row['has_prefix']}",
        flush=True,
    )

    # ---- feature blocks ---------------------------------------------------
    blocks_raw = {"action": action_matrix(meta, rows)}
    if not args.no_vision:
        getter = repack.EpisodeFeatures(args.task, args.scale)
        offsets = (
            exec_of_row if getattr(getter, "indexes_absolute_frames", True)
            else np.zeros_like(exec_of_row)
        )
        try:
            blocks_raw["vision"] = vision_matrix(
                getter, epis_of_row, frame_of_row, args.horizon, "both", offsets
            )
        except FileNotFoundError as exc:
            print(f"  no vision cache ({exc}); action-only", flush=True)

    fitted = {
        name: Block(name, None if name == "action" else args.pca).fit(
            X[train_mask], seed=args.seed
        )
        for name, X in blocks_raw.items()
    }

    conditions = {"action": ["action"]}
    if "vision" in blocks_raw:
        conditions["vision"] = ["vision"]
        conditions["action+vision"] = ["action", "vision"]

    for name, parts in conditions.items():
        X = np.concatenate([fitted[p].transform(blocks_raw[p]) for p in parts], axis=1)
        codes, _ = kmeans_fit_predict(X[train_mask], X, args.k, args.seed)

        all_c, all_l, all_t = [], [], []
        streams = {}
        for ep in eps:
            lo, hi = meta.rows(ep.epis_idx)
            streams[ep.epis_idx] = codes[lo:hi].tolist()
            c, l = label_mutual_information(streams[ep.epis_idx], ep, meta)
            all_c.extend(c)
            all_l.extend(l)
            all_t.extend([ep.epis_idx in train_set] * len(c))
        acc, maj = label_accuracy(all_c, all_l, all_t)
        row[f"label_acc_{name}"] = acc
        row["label_majority_heldout"] = maj

        if name == "action":
            corpus = [
                [r.symbol for r in runs_with_spans(streams[ep.epis_idx], args.min_span)]
                for ep in train_eps
            ]
            vocab = bpe.train(
                corpus, vocab_size=args.vocab_size,
                min_frequency=args.min_frequency, max_token_length=20,
            )
            bpe.assert_no_self_merges(vocab)
            b = boundary_report(
                {ep.epis_idx: streams[ep.epis_idx] for ep in test_eps},
                {ep.epis_idx: ep for ep in test_eps},
                meta, vocab, args.min_span, args.tolerance,
            )
            # Name the BPE tokens themselves. Every other identity number here is
            # per-frame cluster ids; this is the one that describes what would
            # actually go into the log.
            t = token_report(
                {ep.epis_idx: streams[ep.epis_idx] for ep in eps},
                {ep.epis_idx: ep for ep in eps},
                meta, vocab, train_set, args.min_span,
            )
            row.update(t)
            row.update({
                "runs_f1": b["runs_f1"], "bpe_f1": b["bpe_f1"],
                "bpe_precision": b["bpe_precision"], "bpe_recall": b["bpe_recall"],
                "bpe_tokens_per_episode": b["bpe_tokens_per_episode"],
                "over_segmentation_bpe": b["over_segmentation_bpe"],
                "merges": b["merges"],
            })
        print(f"  {name:14s} label acc {acc:6.1%} (majority {maj:.1%})", flush=True)

    if "runs_f1" in row:
        print(
            f"  boundaries     runs F1 {row['runs_f1']:.3f} -> bpe F1 {row['bpe_f1']:.3f}"
            f"   over-seg {row['over_segmentation_bpe']:.1f}x",
            flush=True,
        )
    if row.get("token_instances"):
        print(
            f"  TOKEN identity {row['token_accuracy']:6.1%} "
            f"(majority {row['token_majority']:.1%}, gain {row['token_gain']:+.1%})  "
            f"purity {row['mean_purity']:.2f}  "
            f"straddling {row['straddle_fraction']:.0%}  "
            f"unnamed {row['unseen_token_rate']:.0%}",
            flush=True,
        )

    out = args.out or str(paths.CACHE_ROOT / "eval" / f"alltasks_{args.task}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(row, fh, indent=2)
    print("  wrote", out, flush=True)


if __name__ == "__main__":
    main()
