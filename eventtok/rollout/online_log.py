"""Build the event log during a rollout, from actions the robot has already executed.

This has to produce the *same* log the offline pipeline would, or the closed-loop
numbers describe a different system than every offline number reported so far. Three
things make that non-trivial, and all three are handled here rather than left to the
caller.

**Causality.** Offline, a transition's code comes from ``delta_actions(t)``, the chunk
of ``k`` actions starting at t -- future actions, available because the episode was
already recorded. Online there is no future, so a chunk is only coded once its ``k``
actions have all been executed, which means the code for the chunk starting at t
appears at time t+k. The log at time t therefore contains only spans that closed at or
before t. That is exactly ``EventLog.prefix_at``'s rule, and conditioning on anything
more is the leak that would inflate every counting result.

**The same normalisation.** Codes come from k-means over ``delta_actions / action_scale``
with dead dimensions zeroed. The scale is a property of the training task and is passed
in rather than recomputed, since recomputing it from a partial rollout would give a
different scale every episode.

**The same run/BPE treatment.** Runs shorter than ``min_span`` are dropped as jitter and
BPE merges are applied greedily over the run symbols, matching ``runs_with_spans`` and
``encode_aligned``. BPE is re-applied to the whole run-symbol sequence each time a run
closes rather than incrementally: merges are not prefix-stable, so an incremental
encoder would drift from the offline one exactly where a merge spans the boundary.
"""

from __future__ import annotations

import numpy as np

from ..bpe.build_vocab import BPEVocab


class OnlineEventLog:
    """Feed it executed actions and states; read the current token prefix."""

    def __init__(
        self,
        centroids: np.ndarray,
        action_scale: np.ndarray,
        vocab: BPEVocab,
        chunk: int = 20,
        min_span: int = 3,
        max_log: int = 64,
    ) -> None:
        self.centroids = np.asarray(centroids, dtype=np.float32)
        self.action_scale = np.asarray(action_scale, dtype=np.float32)
        self.vocab = vocab
        self.chunk = chunk
        self.min_span = min_span
        self.max_log = max_log
        self.reset()

    def reset(self) -> None:
        self._actions: list[np.ndarray] = []
        self._states: list[np.ndarray] = []
        self._codes: list[int] = []
        self._tokens: list[int] = []
        self._dirty = False

    # ------------------------------------------------------------------ input
    def push(self, action: np.ndarray, state: np.ndarray) -> None:
        """Record one executed action and the state it was taken from."""
        self._actions.append(np.asarray(action, dtype=np.float32))
        self._states.append(np.asarray(state, dtype=np.float32))
        # A chunk starting at t is complete once t+chunk actions exist.
        t = len(self._actions) - self.chunk
        if t >= 0:
            self._codes.append(self._code_for(t))
            self._dirty = True

    def _code_for(self, t: int) -> int:
        chunk = np.stack(self._actions[t : t + self.chunk])
        # delta_actions: the first seven dims are relative to the state the chunk
        # started from; the gripper stays absolute. Mirrors TaskMeta.delta_actions.
        chunk = chunk.copy()
        chunk[:, :7] -= self._states[t][:7]
        x = (chunk / self.action_scale).ravel()
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        d = ((x[None, :] - self.centroids) ** 2).sum(-1)
        return int(d.argmin())

    # ------------------------------------------------------------------ output
    def _runs(self) -> list[int]:
        out: list[int] = []
        if not self._codes:
            return out
        start = 0
        for i in range(1, len(self._codes) + 1):
            if i == len(self._codes) or self._codes[i] != self._codes[start]:
                if i - start >= self.min_span:
                    out.append(int(self._codes[start]))
                start = i
        return out

    def tokens(self) -> list[int]:
        """The event tokens closed so far, most recent ``max_log`` kept."""
        if self._dirty:
            runs = self._runs()
            self._tokens = self.vocab.encode_span(runs) if runs else []
            self._dirty = False
        return self._tokens[-self.max_log :]

    def counts(self, size: int | None = None) -> np.ndarray:
        """Occurrence count per vocabulary entry -- the derived-state baseline."""
        n = size if size is not None else self.vocab.size
        v = np.zeros(n, dtype=np.float32)
        for t in self.tokens():
            if 0 <= t < n:
                v[t] += 1.0
        return v

    def __len__(self) -> int:
        return len(self.tokens())


def stable_prefix_encode(
    vocab: BPEVocab, runs: list[int], max_token_length: int
) -> list[int]:
    """BPE tokens that can never be revised by future runs.

    Whole-sequence BPE is not prefix-stable. Measured on SwingXtimes, **70% of
    prefixes disagree with the whole-episode encoding**: appending a run can create an
    adjacency that fires an earlier merge, rewriting tokens that were already emitted.
    A log built that way is not causally realisable -- the identity of a token at time
    t depends on runs that happen after t -- and filtering such a log by span end,
    which is what ``prefix_tokens`` did, does not fix it.

    Encoding per closed span, as ``bpe.streaming.StreamingTokenizer`` does, is stable
    but useless here: a span delimited by code changes contains one repeated symbol, so
    every pair is a self-pair and the guard forbids all merges -- the "BPE 0 merges"
    failure this project already hit once.

    The rule used instead: greedy merges cascade leftward by at most the length of the
    longest token, so a token whose span ends more than ``max_token_length`` run
    symbols before the end of the sequence can no longer change. Emit those, hold the
    rest. The log lags reality by a bounded number of runs and is append-only.
    """
    if not runs:
        return []
    ids = vocab.encode_span(runs)
    if not ids:
        return []
    # Walk the emitted tokens, accumulating how many run symbols each consumed.
    consumed, safe = 0, []
    horizon = len(runs) - max_token_length
    for tok in ids:
        width = len(vocab.decode_token(tok))
        if consumed + width > horizon:
            break
        consumed += width
        safe.append(tok)
    return safe
