"""Precompute the causal event log for every frame of a RoboMME task.

One cache per task: for frame *t* of episode *e*, the event tokens a policy could
legitimately have had at *t*. Precomputed because it is deterministic given the
tokenizer, and because recomputing it inside the data loader would put k-means and BPE
on the training hot path.

Three properties this file is responsible for, each of which has already gone wrong
once in this project:

* **Causality.** A chunk starting at *u* is only coded once its *k* actions have
  executed, so its token cannot appear before *u+k*. Whole-episode BPE followed by
  slicing is *not* causal -- token identities then depend on future runs, and 70% of
  prefixes disagreed with the causal encoding.
* **Bounded length with an exact tally.** The visible window is capped; evicted tokens
  keep their counts in an overflow vector. Plain FIFO would delete the multiplicity
  this project exists to preserve.
* **Fit on train episodes only.** k-means centroids and BPE merges come from the train
  split, so the cache for held-out episodes is not fitted on itself.
"""

from __future__ import annotations

import json

import numpy as np

from .. import paths
from ..bpe import build_vocab as bpe
from ..data import repack
from ..data.index import Episode, RoboMMEIndex
from ..data.meta import TaskMeta
from ..eval.bpe_boundaries import runs_with_spans
from ..models.streams import build_streams
from ..rollout.online_log import causal_prefix_table, overflow_at_time, tokens_at_time


def cache_path(task: str, tag: str = "default") -> "paths.Path":
    return paths.CACHE_ROOT / "pi05_events" / f"{task}_{tag}.npz"


def build_task(
    task: str,
    *,
    tokens: str = "action+vision",
    k: int = 16,
    chunk: int = 20,
    min_span: int = 3,
    min_frequency: int = 10,
    max_token_length: int = 4,
    max_log: int = 64,
    scale: str = "2x2",
    seed: int = 0,
    train_frac: float = 0.5,
    index: RoboMMEIndex | None = None,
) -> dict:
    """Build the per-frame log cache for one task.

    Returns a dict of arrays keyed for ``np.savez``: for every (episode, frame) a
    padded token row, its true length, and the overflow tally.
    """
    index = index or RoboMMEIndex()
    eps = index.by_task(task)
    meta = TaskMeta(task)
    feats = repack.EpisodeFeatures(task, scale) if tokens == "action+vision" else None

    order = np.random.default_rng(seed).permutation(len(eps))
    cut = int(len(eps) * train_frac)
    train_eps = [eps[i] for i in order[:cut]]

    streams = build_streams(
        meta, train_eps, eps, tokens=tokens, k=k, seed=seed,
        features=feats, horizon=chunk,
    )
    corpus = [
        [r.symbol for r in runs_with_spans(streams[e.epis_idx], min_span)]
        for e in train_eps
    ]
    vocab = bpe.train(
        corpus, vocab_size=256, min_frequency=min_frequency,
        max_token_length=max_token_length,
    )
    # Symbol space must cover BPE ids; PAD sits one past the end.
    n_symbols = vocab.size
    pad_id = n_symbols

    rows_tok, rows_len, rows_ovf, rows_key = [], [], [], []
    for ep in eps:
        lo, hi = meta.rows(ep.epis_idx)
        runs = runs_with_spans(streams[ep.epis_idx], min_span)
        closed_at, tokens_at = causal_prefix_table(vocab, runs)
        # See joint.build_joint: row t's code uses the frame at t+chunk, and a run is
        # only known to have ended once the differing row after it is coded. Revealing
        # at `end` hands the policy a token computed from a frame it has not seen.
        closed_at = [c + chunk for c in closed_at]
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

    return {
        "tokens": np.stack(rows_tok),
        "lengths": np.asarray(rows_len, dtype=np.int16),
        "overflow": np.stack(rows_ovf).astype(np.float16),
        "keys": np.asarray(rows_key, dtype=np.int32),
        "meta": np.frombuffer(
            json.dumps({
                "task": task, "tokens": tokens, "k": k, "chunk": chunk,
                "min_span": min_span, "min_frequency": min_frequency,
                "max_token_length": max_token_length, "max_log": max_log,
                "scale": scale, "seed": seed, "train_frac": train_frac,
                "n_symbols": n_symbols, "pad_id": pad_id,
                "train_episodes": sorted(int(e.epis_idx) for e in train_eps),
            }).encode(),
            dtype=np.uint8,
        ),
    }


class EventLogCache:
    """Read side: ``cache[epis_idx, frame] -> (tokens, length, overflow)``."""

    def __init__(self, task: str, tag: str = "default") -> None:
        path = cache_path(task, tag)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} missing; run scripts/build_pi05_events.py --task {task}"
            )
        with np.load(path) as d:
            self.tokens = d["tokens"]
            self.lengths = d["lengths"]
            self.overflow = d["overflow"]
            keys = d["keys"]
            self.meta = json.loads(bytes(d["meta"]).decode())
        self._row = {(int(e), int(t)): i for i, (e, t) in enumerate(keys)}
        self.pad_id = int(self.meta["pad_id"])
        self.n_symbols = int(self.meta["n_symbols"])
        self.max_log = int(self.meta["max_log"])

    def __getitem__(self, key: tuple[int, int]):
        i = self._row[key]
        return self.tokens[i], int(self.lengths[i]), self.overflow[i]

    def get(self, epis_idx: int, frame: int):
        """Same, but clamps a frame past the end rather than raising.

        A rollout can run longer than the recorded episode; the last log is the right
        answer there, not a crash.
        """
        key = (int(epis_idx), int(frame))
        if key in self._row:
            return self[key]
        frames = [t for (e, t) in self._row if e == int(epis_idx)]
        if not frames:
            raise KeyError(f"episode {epis_idx} not in cache")
        return self[(int(epis_idx), max(f for f in frames if f <= frame) if
                     any(f <= frame for f in frames) else min(frames))]

    def is_train_episode(self, epis_idx: int) -> bool:
        return int(epis_idx) in set(self.meta["train_episodes"])
