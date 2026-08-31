"""Rebuild the event log during a rollout, from what the policy did and saw.

Training reads the log from a cache. A rollout cannot: the actions are the policy's own,
so the log has to be recomputed from them, with the *same* tokenizer. Anything else and
the policy is tested on a different quantisation of a different signal than it trained
on -- which is a silent failure, because a log built by the wrong rules still looks like
a log.

Three alignments this file exists to hold, each of which the offline pipeline learned the
hard way:

**Frame rate.** ``infer`` is called once per 16 executed steps, but the training cache has
one code per *frame*. Coding one chunk per call would give a sixteenth of the codes, and
a run-length collapse with ``min_span=3`` would then need 48 frames of unchanging
behaviour to emit anything -- the log would be nearly always empty at test time and
routinely populated at training time. So the action chunk for row t is reconstructed from
the executed actions t..t+19, which is what ``OnlineEventLog`` does and what
``test_online_codes_match_offline`` checks against the offline stream.

**Causality.** Row t needs executed actions through t+19 and the visual feature at t+20
(the vision block is ``[feat_t, feat_{t+20} - feat_t]``). Both arrive at t+20, and the
training cache also only reveals row t's token at t+20. The two delays coincide, so
nothing has to be introduced or removed to match.

**Prefix stability.** BPE merges are not prefix stable, so tokens are emitted through
``stable_prefix_encode`` rather than by re-encoding the whole run sequence. Skipping this
is exactly the bug that made the rollout log disagree with the training cache.
"""

from __future__ import annotations

import json

import numpy as np

from ..bpe.build_vocab import BPEVocab
from ..models.multimodal import MultimodalTokenizer
from ..rollout.online_log import stable_prefix_encode, vocab_horizon
from .tokens import cache_path


def tokenizer_path(tag: str = "joint_k64"):
    return cache_path("x", tag).parent / f"tokenizer_{tag}.npz"


class EventTokenizerBundle:
    """The fitted tokenizer, BPE vocabulary and per-task action scales, from disk."""

    def __init__(self, tag: str = "joint_k64") -> None:
        path = tokenizer_path(tag)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} missing; rebuild with build_pi05_joint (it writes the "
                "tokenizer beside the caches)"
            )
        with np.load(path, allow_pickle=True) as d:
            sd = {k: d[k] for k in d.files}
        self.tag = tag
        self.tok = MultimodalTokenizer.from_state_dict(sd)
        b = json.loads(bytes(sd["bpe"]).decode())
        self.vocab = BPEVocab(
            merges=[(tuple(x), tuple(y)) for x, y in b["merges"]],
            tokens=[tuple(t) for t in b["tokens"]],
            min_frequency=b["min_frequency"],
            max_token_length=b["max_token_length"],
        )
        self.horizon_runs = vocab_horizon(self.vocab)
        self.chunk = int(sd["chunk"])
        self.min_span = int(sd["min_span"])
        self.max_log = int(sd["max_log"])
        self.n_symbols = self.vocab.size
        self.pad_id = self.vocab.size
        self._action_scale = {
            k.split(".", 1)[1]: sd[k] for k in sd if k.startswith("action_scale.")
        }

    def action_scale(self, task: str) -> np.ndarray:
        if task not in self._action_scale:
            raise KeyError(
                f"no action scale for {task!r}; the tokenizer was fitted on "
                f"{sorted(self._action_scale)}"
            )
        return self._action_scale[task]

    def code(self, action_rows: np.ndarray, vision_rows: np.ndarray) -> np.ndarray:
        """Nearest centroid per row. ``action_rows`` (n,160), ``vision_rows`` (n,16384)."""
        raw = {"action": np.asarray(action_rows, dtype=np.float32)}
        if self.tok.has_vision:
            raw["vision"] = np.asarray(vision_rows, dtype=np.float32)
        elif vision_rows is not None and len(np.shape(vision_rows)):
            raise ValueError("tokenizer is action-only but vision features were passed")
        return self.tok.encode_raw(raw)


class RolloutEventLog:
    """Append-only causal log for one episode of closed-loop control.

    Feed it every executed step. It codes a row as soon as the data that row needs has
    arrived, collapses runs, and emits only the BPE tokens that future runs cannot
    revise.
    """

    def __init__(self, bundle: EventTokenizerBundle, task: str) -> None:
        self.b = bundle
        self.task = task
        self.scale = bundle.action_scale(task)
        self.reset()

    def reset(self) -> None:
        self._actions: list[np.ndarray] = []
        self._states: list[np.ndarray] = []
        self._vision: list[np.ndarray] = []
        self._codes: list[int] = []
        self._tokens: list[int] = []
        self._dirty = False
        self._final = False

    def end_episode(self) -> None:
        """Close the trailing run. Only for comparing a finished episode offline."""
        self._final = True
        self._dirty = True

    # ------------------------------------------------------------------- input
    def push(self, action: np.ndarray, state: np.ndarray, vision: np.ndarray) -> None:
        """One executed step: the action taken, the state it was taken from, the frame.

        ``vision`` is the flattened per-frame visual feature -- ``image_emb_2x2`` for the
        base view, the same quantity the preprocessed dataset stores and the same one
        ``MemoryBuffer`` computes at inference time.
        """
        self._actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
        self._states.append(np.asarray(state, dtype=np.float32).reshape(-1))
        self._vision.append(np.asarray(vision, dtype=np.float32).reshape(-1))
        self._close_ready()

    def _close_ready(self) -> None:
        """Code every row whose chunk has executed and whose t+chunk frame has arrived."""
        k = self.b.chunk
        n = len(self._actions)
        # Row t needs actions t..t+k-1 and the frames at t and t+k, so it can close only
        # once index t+k exists -- that is, when n >= t+k+1.
        while len(self._codes) + k + 1 <= n:
            t = len(self._codes)
            chunk = np.stack(self._actions[t : t + k])
            # delta_actions: first seven dims relative to the state the chunk started
            # from, gripper absolute. Mirrors TaskMeta.delta_actions.
            chunk = chunk.copy()
            chunk[:, :7] -= self._states[t][:7]
            a = np.nan_to_num((chunk / self.scale).ravel(), nan=0.0,
                              posinf=0.0, neginf=0.0)[None, :]
            f0, f1 = self._vision[t], self._vision[t + k]
            v = np.concatenate([f0, f1 - f0])[None, :]
            self._codes.append(int(self.b.code(a, v)[0]))
            self._dirty = True

    # ------------------------------------------------------------------ output
    def _runs(self, final: bool = False) -> list[int]:
        """Run symbols for the codes so far.

        The trailing run is **not** closed unless ``final``. Offline, ``runs_with_spans``
        runs over a finished episode, so its last run ends at the episode end; mid-episode
        the log contains only runs that a *different* following code terminated. Closing
        the open tail online emits a run one boundary early and the log then runs ahead of
        the training cache -- measured as a 28% frame disagreement that no uniform lag
        could explain.
        """
        out: list[int] = []
        if not self._codes:
            return out
        start = 0
        stop = len(self._codes) + 1 if final else len(self._codes)
        for i in range(1, stop):
            if i == len(self._codes) or self._codes[i] != self._codes[start]:
                if i - start >= self.b.min_span:
                    out.append(int(self._codes[start]))
                start = i
        return out

    def _refresh(self) -> None:
        if not self._dirty:
            return
        runs = self._runs(final=self._final)
        self._tokens = (
            stable_prefix_encode(self.b.vocab, runs, self.b.horizon_runs)
            if runs else []
        )
        self._dirty = False

    def tokens(self) -> list[int]:
        self._refresh()
        return self._tokens[-self.b.max_log :]

    def overflow(self) -> np.ndarray:
        """Eviction tally. Non-zero once a rollout runs long enough to exceed the window.

        A failing rollout is exactly where this matters: an oscillating policy on a
        1300-step episode produces far more tokens than the budget, and plain FIFO would
        delete the multiplicity the method exists to preserve.
        """
        self._refresh()
        v = np.zeros(self.b.n_symbols, dtype=np.float32)
        for t in self._tokens[: max(len(self._tokens) - self.b.max_log, 0)]:
            if 0 <= t < self.b.n_symbols:
                v[t] += 1.0
        return v

    def padded(self) -> tuple[np.ndarray, int, np.ndarray]:
        """``(tokens, length, overflow)`` in the shape the memory features expect."""
        toks = self.tokens()
        row = np.full(self.b.max_log, self.b.pad_id, dtype=np.int64)
        row[: len(toks)] = toks
        return row, len(toks), self.overflow()

    def __len__(self) -> int:
        return len(self.tokens())
