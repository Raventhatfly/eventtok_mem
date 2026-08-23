"""Train the Diffusion Policy with the event log as memory, and run the controls.

    python -m eventtok.scripts.train_diffusion_policy --task PickXtimes

Same split, same log construction and same controls as the behaviour-cloning probe in
``eventtok/consume/probe.py``; only the policy class changes. Reporting DDIM-sampled
action L1 rather than the denoising loss, because the denoising loss is not comparable
across conditioning setups -- a model conditioned on more information faces an easier
noise-prediction problem at every timestep whether or not it produces better actions.

Two axes:

  memory      none / log / wrong / shuffled / count -- as before, with ``wrong`` the
              control that decides whether the content is being read.
  cond        crossattn (log stays a sequence) vs global (everything pooled into one
              vector, the standard DP recipe). The plan asserts pooling destroys the
              count; running both is how that gets tested rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .. import paths
from ..bpe import build_vocab as bpe
from ..consume.diffusion import DiffusionPolicy
from ..consume.probe import build_logs, prefix_tokens
from ..data import repack
from ..data.index import RoboMMEIndex
from ..data.meta import TaskMeta
from ..eval.bpe_boundaries import runs_with_spans
from ..models.kmeans import KMeansTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickXtimes")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--memories", nargs="+", default=["none", "log", "wrong", "count"])
    ap.add_argument("--cond", nargs="+", default=["crossattn", "global"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--diffusion-steps", type=int, default=100)
    ap.add_argument("--sample-steps", type=int, default=20)
    ap.add_argument("--max-log", type=int, default=64)
    ap.add_argument("--min-span", type=int, default=3)
    ap.add_argument("--min-frequency", type=int, default=10)
    ap.add_argument("--scale", default="2x2")
    ap.add_argument("--seed", type=int, default=0)
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
    train_ids = {e.epis_idx for e in train_eps}

    km = KMeansTokenizer(args.k, seed=args.seed).fit(meta, train_eps)
    streams = {e.epis_idx: km.stream_for_episode(meta, e) for e in eps}
    corpus = [[r.symbol for r in runs_with_spans(streams[e.epis_idx], args.min_span)]
              for e in train_eps]
    vocab = bpe.train(corpus, vocab_size=256, min_frequency=args.min_frequency,
                      max_token_length=20)
    logs = build_logs(streams, vocab, args.min_span)

    samples = []
    for ep in eps:
        lo, hi = meta.rows(ep.epis_idx)
        for t in range(hi - lo):
            samples.append((ep.epis_idx, t, lo + t, ep.exec_start + t))
    epis_all = np.array([s[0] for s in samples])
    t_all = np.array([s[1] for s in samples])
    row_all = np.array([s[2] for s in samples])
    frame_all = np.array([s[3] for s in samples])
    is_train = np.array([e in train_ids for e in epis_all])

    scale = meta.action_scale
    Y = np.stack([(meta.delta_actions(int(r)) / scale)[: args.chunk] for r in row_all])
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    V = np.stack([np.asarray(feats[int(e)][int(f)], dtype=np.float32)
                  for e, f in zip(epis_all, frame_all)])
    S = meta.state[row_all].astype(np.float32)
    ref_mean = float(np.abs(Y[~is_train] - Y[is_train].mean(0, keepdims=True)).mean())
    print(f"{args.task}: {len(samples)} transitions, vocab {vocab.size}, "
          f"mean log {np.mean([len(v) for v in logs.values()]):.1f} tokens, "
          f"reference L1 (train mean) {ref_mean:.4f}", flush=True)

    ep_ids = [e.epis_idx for e in eps]
    other = {e: [o for o in ep_ids if o != e][int(rng.integers(len(ep_ids) - 1))]
             for e in ep_ids}

    def memory_arrays(memory: str):
        toks = np.full((len(samples), args.max_log), vocab.size, dtype=np.int64)
        lens = np.zeros(len(samples), dtype=np.int64)
        counts = np.zeros((len(samples), vocab.size), dtype=np.float32)
        if memory == "none":
            return toks, lens, counts
        for i, (e, t, _, _) in enumerate(samples):
            src = other[e] if memory == "wrong" else e
            p = prefix_tokens(logs[src], t, args.max_log)
            if memory == "shuffled" and len(p) > 1:
                p = list(rng.permutation(p))
            lens[i] = len(p)
            if p:
                toks[i, : len(p)] = p
                for tok in p:
                    counts[i, tok] += 1.0
        return toks, lens, counts

    tr, te = np.flatnonzero(is_train), np.flatnonzero(~is_train)
    results = {}
    for cond in args.cond:
        for memory in args.memories:
            key = f"{memory}|{cond}"
            torch.manual_seed(args.seed)
            toks, lens, counts = memory_arrays(memory)
            model = DiffusionPolicy(
                action_dim=Y.shape[-1], k=args.chunk, d_feat=V.shape[-1],
                n_vis=V.shape[1], state_dim=S.shape[-1], vocab=vocab.size,
                d_model=args.d_model, n_layers=args.layers, max_log=args.max_log,
                memory=memory, cond=cond, n_steps=args.diffusion_steps,
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            tt = lambda a, idx: torch.from_numpy(a[idx]).to(device)

            for _ in range(args.epochs):
                model.train()
                perm = rng.permutation(tr)
                for b in range(0, len(perm), args.batch):
                    idx = perm[b : b + args.batch]
                    loss = model.loss(tt(Y, idx), tt(V, idx), tt(S, idx),
                                      tt(toks, idx), tt(lens, idx), tt(counts, idx))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

            model.eval()
            errs = np.zeros(len(te), dtype=np.float32)
            for b in range(0, len(te), 512):
                idx = te[b : b + 512]
                pred = model.sample(tt(V, idx), tt(S, idx), tt(toks, idx),
                                    tt(lens, idx), tt(counts, idx),
                                    steps=args.sample_steps)
                errs[b : b + len(idx)] = (
                    (pred - tt(Y, idx)).abs().mean(dim=(1, 2)).cpu().numpy()
                )
            results[key] = {"sampled_l1": float(errs.mean()),
                            "final_denoise_loss": float(loss.item())}
            print(f"  {key:22s} sampled L1 {errs.mean():.4f}", flush=True)

    print("\n  memory effect, within each conditioning mode:")
    for cond in args.cond:
        base = results.get(f"none|{cond}", {}).get("sampled_l1")
        if base is None:
            continue
        print(f"    {cond}:")
        for memory in args.memories:
            r = results.get(f"{memory}|{cond}")
            if r is None or memory == "none":
                continue
            print(f"      {memory:9s} {r['sampled_l1'] / base - 1:+7.1%} vs none")
        lg = results.get(f"log|{cond}", {}).get("sampled_l1")
        wr = results.get(f"wrong|{cond}", {}).get("sampled_l1")
        if lg and wr and base - lg > 1e-4:
            print(f"      content share of the memory benefit: {(wr - lg) / (base - lg):.0%}")
        elif lg and base - lg <= 1e-4:
            print(f"      memory does not help here; content share undefined")

    if len(args.cond) > 1 and all(f"log|{c}" in results for c in args.cond):
        a, b = results["log|crossattn"]["sampled_l1"], results["log|global"]["sampled_l1"]
        print(f"\n  crossattn vs pooled global_cond, with the log: {a - b:+.4f} "
              f"({(a / b - 1):+.1%})")
        print("  " + ("keeping the log a sequence beats pooling it, which is what the "
                      "plan predicted" if a < b else
                      "pooling is no worse here, so the plan's objection to global_cond "
                      "is not supported on this task"))

    out = args.out or str(paths.CACHE_ROOT / "eval" / f"dp_{args.task}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"task": args.task, "reference_mean_L1": ref_mean,
                   "results": results}, fh, indent=2)
    print("\n  wrote", out)


if __name__ == "__main__":
    main()
