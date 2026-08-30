"""Turn a cached event log into the arrays pi0.5's memory slot expects.

The integration deliberately reuses robomme_policy_learning's *existing* perceptual
memory path rather than adding a new one. That path is generic: ``PerceptualMemory``
concatenates ``[img_emb, silu(pos_proj(pos)), silu(state_proj(state))]`` and applies one
linear map to the LLM width, then ``embed_prefix`` prepends the result with a mask. It
contains no MovieChat logic -- the merging happens on the data side upstream -- so the
same wiring carries our log unchanged.

Riding that path buys the comparison we actually need: MovieChat, token-dropping and
event memory then differ *only* in what fills the slots. Any difference in success rate
is attributable to the memory content, not to a different encoder or attention pattern.

The encoding:

``static_image_emb``  one-hot over the event vocabulary. A one-hot times a linear map is
    an embedding-table lookup, so ``encoder_static``'s rows *are* the learned event
    embeddings -- no new module, and no risk of the token id being read as a number the
    way it would be if we spelled the log into the text prompt.
``static_pos_emb``    sinusoid of the slot index. Required, not optional: the prefix is
    bidirectional, so without it the slots are a set and "swung three times" is
    indistinguishable from "swung once".
``static_state_emb``  log1p of the eviction tally, per vocabulary entry. The window is
    bounded; this is what keeps the count exact once it overflows.
``static_mask``       which slots hold a real token.
"""

from __future__ import annotations

import numpy as np


def slot_posemb(n_slots: int, dim: int = 64) -> np.ndarray:
    """Sinusoidal encoding of slot index, ``(n_slots, dim)``.

    Fixed rather than learned so that a log longer than anything seen in training still
    gets a well-defined position -- the extrapolation-to-unseen-N experiment depends on
    positions past the training range behaving sensibly.
    """
    if dim % 2:
        raise ValueError(f"dim must be even, got {dim}")
    pos = np.arange(n_slots, dtype=np.float32)[:, None]
    omega = np.exp(-np.log(10_000.0) * np.arange(dim // 2, dtype=np.float32) / (dim // 2))
    ang = pos * omega[None, :]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=-1).astype(np.float32)


class EventMemoryFeatures:
    """Cached log row -> the four ``static_*`` arrays.

    Args:
        n_symbols: event vocabulary size. Fixed across tasks when the caches were built
            jointly; a per-task vocabulary makes token 7 mean different things in
            different tasks and must not be used to train one multi-task policy.
        max_log: number of memory slots (the ``budget`` the history config declares).
        pos_dim: width of the slot sinusoid.
    """

    def __init__(self, n_symbols: int, max_log: int, pos_dim: int = 64) -> None:
        self.n_symbols = int(n_symbols)
        self.max_log = int(max_log)
        self.pos = slot_posemb(self.max_log, pos_dim)

    @property
    def dims(self) -> dict[str, int]:
        return {"img": self.n_symbols, "pos": self.pos.shape[1], "state": self.n_symbols}

    def __call__(self, tokens: np.ndarray, length: int, overflow: np.ndarray) -> dict:
        """``tokens`` is the padded row, ``length`` how much of it is real."""
        onehot = np.zeros((self.max_log, self.n_symbols), dtype=np.float32)
        length = int(min(length, self.max_log))
        if length:
            ids = np.asarray(tokens[:length], dtype=np.int64)
            if ids.max(initial=-1) >= self.n_symbols:
                raise ValueError(
                    f"token id {ids.max()} >= vocab {self.n_symbols}; the cache and the "
                    "history config disagree about the vocabulary"
                )
            onehot[np.arange(length), ids] = 1.0

        mask = np.zeros(self.max_log, dtype=bool)
        mask[:length] = True

        # Broadcast over slots: the tally is a property of the log, not of one slot, and
        # the encoder has no other way to see it.
        tally = np.log1p(np.asarray(overflow, dtype=np.float32))
        state = np.broadcast_to(tally, (self.max_log, self.n_symbols)).astype(np.float32)

        return {
            "static_image_emb": onehot,
            "static_pos_emb": self.pos.copy(),
            "static_state_emb": state.copy(),
            "static_mask": mask,
        }

    def blank(self) -> dict:
        """An empty log -- what a policy sees before the first chunk closes."""
        return self(
            np.zeros(self.max_log, dtype=np.int64), 0,
            np.zeros(self.n_symbols, dtype=np.float32),
        )
