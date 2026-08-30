"""One event vocabulary shared by all 16 RoboMME tasks.

RoboMME trains a single policy over every task, so the caches must agree on what a token
id means. Fitting per task and reusing the ids would give one embedding row the job of
being a swing in one task and a peg insertion in another -- the tokenizer would look
fine in the per-task offline numbers and be meaningless to the policy.

Fitting jointly also puts the vocabulary's generality on the line. If the same 16
centroids cannot cover 16 tasks, the per-task label-accuracy result was measuring
per-task overfitting, and that is worth finding out before a training run rather than
after.
"""

from __future__ import annotations

import json

import numpy as np

from ..bpe import build_vocab as bpe
from ..data import repack
from ..data.index import RoboMMEIndex
from ..data.meta import TaskMeta
from ..eval.bpe_boundaries import runs_with_spans
from ..models.multimodal import MultimodalTokenizer
from ..rollout.online_log import causal_prefix_table, overflow_at_time, tokens_at_time

TASKS = [
    "BinFill", "ButtonUnmask", "ButtonUnmaskSwap", "InsertPeg", "MoveCube",
    "PatternLock", "PickHighlight", "PickXtimes", "RouteStick", "StopCube",
    "SwingXtimes", "VideoPlaceButton", "VideoPlaceOrder", "VideoRepick",
    "VideoUnmask", "VideoUnmaskSwap",
]


def _split(eps, seed: int, train_frac: float):
    order = np.random.default_rng(seed).permutation(len(eps))
    cut = int(len(eps) * train_frac)
    return [eps[i] for i in order[:cut]], [eps[i] for i in order[cut:]]


def build_joint(
    tasks: list[str] | None = None,
    *,
    k: int = 16,
    chunk: int = 20,
    min_span: int = 3,
    min_frequency: int = 10,
    max_token_length: int = 4,
    max_log: int = 64,
    scale: str = "2x2",
    seed: int = 0,
    train_frac: float = 0.5,
    fit_episodes: int = 12,
    fit_rows: int = 60_000,
    vision_weight: float = 1.0,
    on_task=None,
) -> dict:
    """Fit one tokenizer across ``tasks``, then emit a per-frame cache for each.

    Args:
        fit_episodes: train episodes per task used for the k-means/PCA fit. The full
            train split is used for the BPE corpus; only the continuous fit is
            subsampled, because holding every task's raw vision block at once is ~26 GB.
        fit_rows: total rows kept for the fit, split evenly across tasks.
        on_task: optional ``callback(task, blob)`` called as each task's cache is
            finished, so a long build can write incrementally.

    Returns:
        ``{task: blob}``, each blob in the ``np.savez`` form ``EventLogCache`` reads.
    """
    tasks = list(tasks or TASKS)
    index = RoboMMEIndex()
    rng = np.random.default_rng(seed)

    metas, feats, splits = {}, {}, {}
    for task in tasks:
        metas[task] = TaskMeta(task)
        feats[task] = repack.EpisodeFeatures(task, scale)
        eps = index.by_task(task)
        splits[task] = (eps, *_split(eps, seed, train_frac))

    tok = MultimodalTokenizer(
        n_clusters=k, horizon=chunk, vision_weight=vision_weight, seed=seed
    )

    raws = []
    for task in tasks:
        _, train_eps, _ = splits[task]
        pick = train_eps[: fit_episodes] if len(train_eps) > fit_episodes else train_eps
        raw = tok.raw_for(metas[task], pick, feats[task])
        per = max(1, fit_rows // len(tasks))
        n = len(next(iter(raw.values())))
        keep = rng.choice(n, min(per, n), replace=False)
        raws.append({name: X[keep] for name, X in raw.items()})
        print(f"  fit rows from {task}: {len(keep)}", flush=True)
    tok.fit_raw(raws)
    del raws

    streams, corpus = {}, []
    for task in tasks:
        all_eps, train_eps, _ = splits[task]
        raw = tok.raw_for(metas[task], all_eps, feats[task])
        codes = tok.encode_raw(raw)
        del raw
        s, i = {}, 0
        for ep in all_eps:
            lo, hi = metas[task].rows(ep.epis_idx)
            s[ep.epis_idx] = codes[i : i + (hi - lo)].tolist()
            i += hi - lo
        streams[task] = s
        # sorted, not a set: BPE merge ties break on corpus order, and a run whose
        # vocabulary depends on dict iteration order is not reproducible.
        corpus.extend(
            [r.symbol for r in runs_with_spans(s[e.epis_idx], min_span)]
            for e in sorted(train_eps, key=lambda e: e.epis_idx)
        )
        print(f"  coded {task}: {len(all_eps)} episodes", flush=True)

    vocab = bpe.train(
        corpus, vocab_size=256, min_frequency=min_frequency,
        max_token_length=max_token_length,
    )
    n_symbols, pad_id = vocab.size, vocab.size
    print(f"joint vocab: {n_symbols} symbols over {len(tasks)} tasks", flush=True)

    shared = {
        "tokens": "action+vision", "k": k, "chunk": chunk, "min_span": min_span,
        "min_frequency": min_frequency, "max_token_length": max_token_length,
        "max_log": max_log, "scale": scale, "seed": seed, "train_frac": train_frac,
        "n_symbols": n_symbols, "pad_id": pad_id, "joint_tasks": tasks,
        "vision_weight": vision_weight,
    }

    out = {}
    for task in tasks:
        all_eps, train_eps, _ = splits[task]
        rows_tok, rows_len, rows_ovf, rows_key = [], [], [], []
        for ep in all_eps:
            lo, hi = metas[task].rows(ep.epis_idx)
            runs = runs_with_spans(streams[task][ep.epis_idx], min_span)
            closed_at, tokens_at = causal_prefix_table(vocab, runs)
            for t in range(hi - lo):
                p = tokens_at_time(closed_at, tokens_at, t, max_log)
                row = np.full(max_log, pad_id, dtype=np.int16)
                if p:
                    row[: len(p)] = p
                rows_tok.append(row)
                rows_len.append(len(p))
                rows_ovf.append(
                    overflow_at_time(closed_at, tokens_at, t, max_log, n_symbols)
                )
                rows_key.append((ep.epis_idx, t))
        blob = {
            "tokens": np.stack(rows_tok),
            "lengths": np.asarray(rows_len, dtype=np.int16),
            "overflow": np.stack(rows_ovf).astype(np.float16),
            "keys": np.asarray(rows_key, dtype=np.int32),
            "meta": np.frombuffer(
                json.dumps({
                    **shared, "task": task,
                    "train_episodes": sorted(int(e.epis_idx) for e in train_eps),
                }).encode(),
                dtype=np.uint8,
            ),
        }
        lens = blob["lengths"].astype(int)
        print(
            f"  {task}: {len(lens)} frames, log {lens.mean():.1f} mean / {lens.max()} max, "
            f"empty {100 * (lens == 0).mean():.0f}%",
            flush=True,
        )
        if on_task is not None:
            on_task(task, blob)
        out[task] = blob
    return out
