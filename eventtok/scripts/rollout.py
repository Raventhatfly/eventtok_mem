"""Closed-loop rollouts: does the event log raise the SUCCESS RATE?

    python -m eventtok.scripts.rollout --task SwingXtimes --episodes 20

Every other number in this project is action prediction on recorded episodes -- open
loop, scored against a demonstrator from the demonstrator's own state distribution. This
is the measurement all of it was a proxy for, and the ladder result makes it the only
one that can still favour the design: raw action history beats event tokens by ~55
points at predicting actions, and the argument for tokens is that a compact countable
memory matters for *acting* in a way a 32-step buffer does not. That argument is only
testable here.

Runs on H100 or RTX Pro 6000 -- verified; A100 fails SAPIEN device creation with
ErrorInitializationFailed, and the whole rollout path was written off earlier on the
strength of one A100 failure.

Vision is DINOv2 rather than the cached SigLIP because the simulator lives in the torch
env and pi0.5's SigLIP tokenizer is Flax in a different one. The encoder comparison
found the two within a couple of points at usable codebook sizes, so this costs little.

The policy is trained here rather than loaded: the diffusion trainer never saved
checkpoints, and plumbing that in is more code than retraining.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

BENCH_SRC = ("/n/home04/wfy/repos/robomme_policy_learning/"
             "third_party/robomme_benchmark/src")


def obs_state(obs) -> np.ndarray:
    """The 8-dim state in the dataset's convention: 7 joint angles + gripper.

    ``build_robomme_dataset`` builds it as ``concat(joint_state, gripper_state[:1])``,
    and the simulator exposes those as ``joint_state_list`` and ``gripper_state_list``.
    There is **no** ``ee_pose`` key -- an earlier version read one with a zeros default,
    so the state was silently all zeros, the delta-to-absolute conversion added nothing,
    and every episode died on the first step with an IK error.
    """
    j = np.asarray(obs["joint_state_list"][-1], dtype=np.float32).reshape(-1)[:7]
    g = np.asarray(obs["gripper_state_list"][-1], dtype=np.float32).reshape(-1)[:1]
    return np.concatenate([j, g]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="SwingXtimes")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--conditions", nargs="+", default=["none", "log", "wrong"])
    ap.add_argument("--tokens", default="action+vision",
                    choices=["action", "action+vision"])
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-span", type=int, default=3)
    ap.add_argument("--min-frequency", type=int, default=10)
    ap.add_argument("--max-token-length", type=int, default=4)
    ap.add_argument("--max-log", type=int, default=64)
    ap.add_argument("--action-space", default="joint_angle",
                    help="joint_angle matches the dataset: its actions are\n                         7 joint targets + gripper, with the action-space\n                         bounds being Franka joint limits.")
    ap.add_argument("--max-steps", type=int, default=1300)
    ap.add_argument("--sample-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if BENCH_SRC not in sys.path:
        sys.path.insert(0, BENCH_SRC)

    from .. import paths
    from ..bpe import build_vocab as bpe
    from ..consume.diffusion import DiffusionPolicy
    from ..data import dino
    from ..data.index import RoboMMEIndex
    from ..data.meta import TaskMeta
    from ..eval.bpe_boundaries import runs_with_spans
    from ..models.streams import build_streams
    from ..rollout.online_log import OnlineEventLog

    paths.check_root()
    device = torch.device("cuda")
    index = RoboMMEIndex()
    eps = index.by_task(args.task)
    meta = TaskMeta(args.task)
    feats = dino.EpisodeDinoFeatures(args.task, "dinov2l", "2x2", "image")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(eps))
    train_eps = [eps[i] for i in order[: len(eps) // 2]]

    # --- tokenizer and vocabulary, fitted offline on the train half ------------
    streams = build_streams(meta, train_eps, eps, tokens=args.tokens, k=args.k,
                            seed=args.seed, features=feats, horizon=args.chunk)
    corpus = [[r.symbol for r in runs_with_spans(streams[e.epis_idx], args.min_span)]
              for e in train_eps]
    vocab = bpe.train(corpus, vocab_size=256, min_frequency=args.min_frequency,
                      max_token_length=args.max_token_length)
    from ..models.multimodal import MultimodalTokenizer  # noqa: F401  (documented below)

    # Centroids for the online log. For action+vision tokens the online encoder cannot
    # reproduce the vision block without the PCA basis, so the log uses the action-only
    # centroids at rollout time and this is recorded as a limitation rather than hidden:
    # the rollout log is action-tokenised even when the offline analysis was multimodal.
    from ..models.kmeans import KMeansTokenizer
    km_online = KMeansTokenizer(args.k, seed=args.seed).fit(meta, train_eps)

    SYMBOLS = max(vocab.size, args.k) + 1
    scale = meta.action_scale

    # --- offline training data ------------------------------------------------
    from ..rollout.online_log import causal_prefix_table, tokens_at_time
    logs = {i: causal_prefix_table(vocab, runs_with_spans(c, args.min_span))
            for i, c in streams.items()}
    rows, V, S, Y, TOK, LEN = [], [], [], [], [], []
    ep_ids = [e.epis_idx for e in eps]
    other = {e: [o for o in ep_ids if o != e][int(rng.integers(len(ep_ids) - 1))]
             for e in ep_ids}
    for ep in train_eps:
        lo, hi = meta.rows(ep.epis_idx)
        arr = feats[ep.epis_idx]
        ca, ta = logs[ep.epis_idx]
        for t in range(hi - lo):
            V.append(np.asarray(arr[t], dtype=np.float32))
            S.append(meta.state[lo + t].astype(np.float32))
            y = (meta.delta_actions(lo + t) / scale)[: args.chunk]
            Y.append(np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
            p = tokens_at_time(ca, ta, t, args.max_log)
            row = np.full(args.max_log, SYMBOLS, dtype=np.int64)
            row[: len(p)] = p
            TOK.append(row)
            LEN.append(len(p))
    V, S, Y = np.stack(V), np.stack(S), np.stack(Y)
    TOK, LEN = np.stack(TOK), np.asarray(LEN)
    a_lo = Y.reshape(-1, Y.shape[-1]).min(0)
    a_hi = Y.reshape(-1, Y.shape[-1]).max(0)
    span = np.maximum(a_hi - a_lo, 1e-6)
    Yn = np.clip(2.0 * (Y - a_lo) / span - 1.0, -1, 1).astype(np.float32)
    print(f"{args.task}: {len(Y)} train transitions, vocab {vocab.size}, "
          f"tokens={args.tokens}", flush=True)

    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    encoder = dino.Dinov2Encoder("dinov2l", device=device, batch=8)
    results = {}
    for cond in args.conditions:
        torch.manual_seed(args.seed)
        model = DiffusionPolicy(
            action_dim=Y.shape[-1], k=args.chunk, d_feat=V.shape[-1], n_vis=V.shape[1],
            state_dim=S.shape[-1], vocab=SYMBOLS, max_log=args.max_log,
            memory=("log" if cond in ("log", "wrong") else "none"),
            cond="crossattn",
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        tt = lambda a, idx: torch.from_numpy(a[idx]).to(device)
        n = len(Yn)
        for _ in range(args.epochs):
            model.train()
            perm = rng.permutation(n)
            for b in range(0, n, args.batch):
                idx = perm[b : b + args.batch]
                loss = model.loss(tt(Yn, idx), tt(V, idx), tt(S, idx),
                                  tt(TOK, idx), tt(LEN, idx), None)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        model.eval()
        print(f"  [{cond}] trained, final denoise loss {loss.item():.4f}", flush=True)

        builder = BenchmarkEnvBuilder(env_id=args.task, dataset="test",
                                      action_space=args.action_space, max_steps=args.max_steps)
        n_ep = min(args.episodes, builder.get_episode_num())
        successes, outcomes = 0, []
        for e in range(n_ep):
            env = builder.make_env_for_episode(e)
            obs, info = env.reset()
            st0 = obs_state(obs)
            olog = OnlineEventLog(km_online.centroids, scale, vocab, chunk=args.chunk,
                                  min_span=args.min_span, max_log=args.max_log)
            # `wrong` replays another episode's log, so the memory is well-formed and
            # same-distribution but describes a different history.
            wrong_src = None
            if cond == "wrong":
                src = other[eps[e % len(eps)].epis_idx]
                wrong_src = logs[src]
            outcome, steps = "timeout", 0
            while steps < args.max_steps:
                front = np.asarray(obs["front_rgb_list"][-1], dtype=np.uint8)
                f = torch.from_numpy(
                    encoder.encode(front[None], scale="2x2").astype(np.float32)
                ).to(device)
                st = obs_state(obs)
                stt = torch.from_numpy(st[None]).to(device)
                if cond == "wrong":
                    p = tokens_at_time(*wrong_src, steps, args.max_log)
                elif cond == "log":
                    p = olog.tokens()
                else:
                    p = []
                row = np.full((1, args.max_log), SYMBOLS, dtype=np.int64)
                row[0, : len(p)] = p
                tok = torch.from_numpy(row).to(device)
                ln = torch.tensor([len(p)], device=device)
                x = model.sample(f, stt, tok, ln, None, steps=args.sample_steps)
                chunk = x[0].cpu().numpy()
                chunk = (chunk + 1.0) / 2.0 * span + a_lo          # back to std units
                chunk = chunk * scale                              # back to raw delta
                chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
                done = False
                for j in range(args.chunk):
                    a = chunk[j].copy()
                    a[:7] += st[:7]                                # delta -> absolute
                    a = np.clip(a, env.action_space.low, env.action_space.high)
                    obs, r, term, trunc, info = env.step(a.astype(np.float32))
                    olog.push(chunk[j], st)
                    steps += 1
                    if isinstance(info, dict) and info.get("status") == "error":
                        outcome, done = "error", True
                        break
                    if bool(np.asarray(term).any()) or bool(np.asarray(trunc).any()):
                        outcome = info.get("status", "unknown") if isinstance(info, dict) else "unknown"
                        done = True
                        break
                if done:
                    break
            successes += outcome == "success"
            outcomes.append(outcome)
            env.close()
            print(f"    ep {e}: {outcome} ({steps} steps)", flush=True)
        results[cond] = {"episodes": n_ep, "successes": successes,
                         "success_rate": successes / max(n_ep, 1),
                         "outcomes": outcomes}
        print(f"  [{cond}] SUCCESS {successes}/{n_ep} = {successes / max(n_ep,1):.0%}",
              flush=True)

    print("\n  success rate by condition:")
    for c, r in results.items():
        print(f"    {c:6s} {r['success_rate']:.0%}  ({r['successes']}/{r['episodes']})")
    out = args.out or str(paths.CACHE_ROOT / "eval" / f"rollout_{args.task}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"task": args.task, "tokens": args.tokens, "results": results}, fh, indent=2)
    print("  wrote", out)


if __name__ == "__main__":
    main()
