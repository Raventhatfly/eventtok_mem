"""Train the memory probe under every condition and report the contrasts.

    python -m eventtok.scripts.eval_consumption --task PickXtimes

See eventtok/consume/probe.py for what this is and is not: an offline
behaviour-cloning probe, not rollout success, because there is no simulator here.

Reported per condition: held-out action L1, overall and restricted to transitions near
an annotated event boundary, where a memory of what has already happened is most likely
to matter. Two reference points are printed alongside, because an L1 without them is
unreadable: predicting the dataset mean action, and predicting zero.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch import nn

from .. import paths
from ..bpe import build_vocab as bpe
from ..consume.probe import CONDITIONS, MemoryPolicy, build_logs, prefix_tokens
from ..data import repack, subgoals as sg
from ..data.index import RoboMMEIndex
from ..data.meta import TaskMeta
from ..eval.bpe_boundaries import runs_with_spans
from ..models.kmeans import KMeansTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickXtimes")
    ap.add_argument("--k", type=int, default=16, help="k-means codebook")
    ap.add_argument("--chunk", type=int, default=20, help="action chunk length")
    ap.add_argument("--min-span", type=int, default=3)
    ap.add_argument("--min-frequency", type=int, default=10)
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--max-log", type=int, default=64)
    ap.add_argument("--scale", default="2x2")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--near-boundary", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths.check_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = RoboMMEIndex()
    eps = index.by_task(args.task)
    meta = TaskMeta(args.task)
    feats = repack.EpisodeFeatures(args.task, args.scale)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(eps))
    cut = len(eps) // 2
    train_eps = [eps[i] for i in order[:cut]]
    test_eps = [eps[i] for i in order[cut:]]
    train_ids = {e.epis_idx for e in train_eps}

    # --- the event log, built exactly as the rest of the project builds it -----
    km = KMeansTokenizer(args.k, seed=args.seed).fit(meta, train_eps)
    streams = {e.epis_idx: km.stream_for_episode(meta, e) for e in eps}
    corpus = [
        [r.symbol for r in runs_with_spans(streams[e.epis_idx], args.min_span)]
        for e in train_eps
    ]
    vocab = bpe.train(corpus, vocab_size=args.vocab_size,
                      min_frequency=args.min_frequency, max_token_length=20)
    bpe.assert_no_self_merges(vocab)
    logs = build_logs(streams, vocab, args.min_span)
    print(f"{args.task}: {len(eps)} eps, vocab {vocab.size}, "
          f"mean log length {np.mean([len(v) for v in logs.values()]):.1f} tokens",
          flush=True)

    # --- samples ---------------------------------------------------------------
    scale = meta.action_scale
    samples = []
    for ep in eps:
        lo, hi = meta.rows(ep.epis_idx)
        n = hi - lo
        segs = sg.segments_from_track(meta.episode_labels(ep.epis_idx), ep.exec_start)
        bounds = np.array([s.start - ep.exec_start for s in segs if s.start > ep.exec_start])
        arr = feats[ep.epis_idx]
        for t in range(n):
            near = bool(len(bounds) and np.abs(bounds - t).min() <= args.near_boundary)
            samples.append((ep.epis_idx, t, lo + t, ep.exec_start + t, near))
    print(f"  {len(samples)} transitions, "
          f"{sum(1 for s in samples if s[4])} near a boundary", flush=True)

    epis_all = np.array([s[0] for s in samples])
    t_all = np.array([s[1] for s in samples])
    row_all = np.array([s[2] for s in samples])
    frame_all = np.array([s[3] for s in samples])
    near_all = np.array([s[4] for s in samples])
    is_train = np.array([e in train_ids for e in epis_all])

    Y = np.stack([(meta.delta_actions(int(r)) / scale)[: args.chunk] for r in row_all])
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    V = np.stack([np.asarray(feats[int(e)][int(f)], dtype=np.float32)
                  for e, f in zip(epis_all, frame_all)])
    S = meta.state[row_all].astype(np.float32)

    mean_action = Y[is_train].mean(0, keepdims=True)
    ref_mean = float(np.abs(Y[~is_train] - mean_action).mean())
    ref_zero = float(np.abs(Y[~is_train]).mean())
    print(f"  reference L1: predict-train-mean {ref_mean:.4f}, predict-zero {ref_zero:.4f}",
          flush=True)

    # --- memory tensors, per condition -----------------------------------------
    other = {}                      # for `wrong`: a different episode, same task
    ep_ids = [e.epis_idx for e in eps]
    for e in ep_ids:
        pool = [o for o in ep_ids if o != e]
        other[e] = pool[int(rng.integers(len(pool)))]

    def memory_arrays(condition: str):
        toks = np.zeros((len(samples), args.max_log), dtype=np.int64)
        lens = np.zeros(len(samples), dtype=np.int64)
        counts = np.zeros((len(samples), vocab.size), dtype=np.float32)
        pad = vocab.size
        toks[:] = pad
        for i, (e, t, _, _, _) in enumerate(samples):
            if condition == "wrong":
                src, tt = other[e], t
            else:
                src, tt = e, t
            p = prefix_tokens(logs[src], tt, args.max_log)
            if condition == "shuffled" and len(p) > 1:
                p = list(rng.permutation(p))
            lens[i] = len(p)
            if p:
                toks[i, : len(p)] = p
                for tok in p:
                    counts[i, tok] += 1.0
        return toks, lens, counts

    results = {}
    for cond in args.conditions:
        torch.manual_seed(args.seed)
        toks, lens, counts = (
            memory_arrays(cond) if cond != "none"
            else (np.zeros((len(samples), args.max_log), dtype=np.int64),
                  np.zeros(len(samples), dtype=np.int64),
                  np.zeros((len(samples), vocab.size), dtype=np.float32))
        )
        model = MemoryPolicy(
            vocab=vocab.size, d_feat=V.shape[-1], n_vis=V.shape[1],
            state_dim=S.shape[-1], action_dim=Y.shape[-1], k=args.chunk,
            d_model=args.d_model, max_log=args.max_log, condition=cond,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        tr = np.flatnonzero(is_train)
        te = np.flatnonzero(~is_train)
        tt = lambda a, idx: torch.from_numpy(a[idx]).to(device)

        for epoch in range(args.epochs):
            model.train()
            perm = rng.permutation(tr)
            for b in range(0, len(perm), args.batch):
                idx = perm[b : b + args.batch]
                pred = model(tt(V, idx), tt(S, idx), tt(toks, idx),
                             tt(lens, idx), tt(counts, idx))
                loss = nn.functional.l1_loss(pred, tt(Y, idx))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        model.eval()
        errs = np.zeros(len(te), dtype=np.float32)
        with torch.inference_mode():
            for b in range(0, len(te), 1024):
                idx = te[b : b + 1024]
                pred = model(tt(V, idx), tt(S, idx), tt(toks, idx),
                             tt(lens, idx), tt(counts, idx))
                errs[b : b + len(idx)] = (
                    (pred - tt(Y, idx)).abs().mean(dim=(1, 2)).cpu().numpy()
                )
        near = near_all[te]
        results[cond] = {
            "l1": float(errs.mean()),
            "l1_near_boundary": float(errs[near].mean()) if near.any() else float("nan"),
            "l1_elsewhere": float(errs[~near].mean()) if (~near).any() else float("nan"),
            "n_test": int(len(te)),
        }
        r = results[cond]
        print(f"  {cond:9s} L1 {r['l1']:.4f}   near-boundary {r['l1_near_boundary']:.4f}"
              f"   elsewhere {r['l1_elsewhere']:.4f}", flush=True)

    print("\n  contrasts (negative = the first condition predicts better):")
    def delta(a, b, key="l1"):
        if a in results and b in results:
            d = results[a][key] - results[b][key]
            rel = d / max(results[b][key], 1e-9)
            return f"{d:+.4f} ({rel:+.1%})"
        return "n/a"
    for a, b, why in [
        ("log", "none", "does memory help at all"),
        ("log", "wrong", "does the CONTENT matter"),
        ("log", "shuffled", "does order matter"),
        ("log", "count", "does the raw log beat counting"),
    ]:
        print(f"    {a:5s} - {b:9s} {delta(a, b):>22s}   {why}")

    if "log" in results and "wrong" in results:
        gap = results["wrong"]["l1"] - results["log"]["l1"]
        benefit = results["none"]["l1"] - results["log"]["l1"] if "none" in results else None
        print()
        if benefit is not None and benefit <= 1e-4:
            # The share is undefined when memory does not help: the denominator is
            # zero or negative and the ratio explodes. Printing it produced
            # "3596907854% of the total benefit", which is worse than saying nothing.
            print(f"  Memory does NOT help on this task: log {results['log']['l1']:.4f} "
                  f"against none {results['none']['l1']:.4f}. The content share is "
                  f"undefined\n  (no benefit to apportion). A wrong log is still "
                  f"{gap:+.4f} relative to the correct one.")
        elif gap <= 0:
            print("  A WRONG log predicts at least as well as the correct one. The policy\n"
                  "  is not reading the memory content, and any transfer number built on\n"
                  "  this log would be measuring nothing.")
        else:
            print(f"  A wrong log is {gap:+.4f} worse than the correct one, i.e. "
                  f"{gap / benefit:.0%} of the\n  total benefit of having memory at all. "
                  f"The content is being read.")

    payload = {"task": args.task, "k": args.k, "vocab": vocab.size,
               "reference_mean_L1": ref_mean, "reference_zero_L1": ref_zero,
               "results": results}
    out = args.out or str(paths.CACHE_ROOT / "eval" / f"consumption_{args.task}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\n  wrote", out)


if __name__ == "__main__":
    main()
