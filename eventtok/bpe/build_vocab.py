"""BPE over code streams, with the two guards that keep counts intact.

Implemented directly rather than through ``tokenizers``. The guards are the whole
point, and both hold *by construction* here:

**Guard 1 — no self-merges.** A pair ``(A, A)`` is never merged. Without this,
``SWING SWING`` becomes one token and the count is destroyed — and since a
repeated action is exactly what these tasks consist of, that pair is among the
most frequent in the corpus, so plain BPE would merge it almost immediately. This
is the single failure that would invalidate every number in the paper while
leaving the pipeline looking healthy.

**Guard 2 — no merges across event boundaries.** The corpus is a list of *spans*,
one per completed event, so adjacent pairs are only ever counted within a span.
Nothing to enforce: a pair straddling a boundary never exists to be counted.

A consequence worth stating: the spans used for training must come from the same
online boundary detector used at test time, not from ground-truth subgoal
annotations. Otherwise the learned merges do not match the segmentation the model
will actually see.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

Symbol = tuple[int, ...]     # a merged token is the tuple of base codes it covers


@dataclass
class BPEVocab:
    """Learned merges plus the resulting token table."""

    merges: list[tuple[Symbol, Symbol]] = field(default_factory=list)
    # token id -> the base-code tuple it stands for
    tokens: list[Symbol] = field(default_factory=list)
    min_frequency: int = 10
    max_token_length: int = 20

    @property
    def size(self) -> int:
        return len(self.tokens)

    def id_of(self, symbol: Symbol) -> int:
        return self._lookup[symbol]

    def __post_init__(self) -> None:
        self._lookup = {s: i for i, s in enumerate(self.tokens)}

    def rebuild_lookup(self) -> None:
        self._lookup = {s: i for i, s in enumerate(self.tokens)}

    # ------------------------------------------------------------------ apply

    def encode_span(self, span: Sequence[int]) -> list[int]:
        """Apply the merges greedily within one span -> token ids."""
        symbols: list[Symbol] = [(int(c),) for c in span]
        for left, right in self.merges:
            i = 0
            out: list[Symbol] = []
            while i < len(symbols):
                if (
                    i + 1 < len(symbols)
                    and symbols[i] == left
                    and symbols[i + 1] == right
                ):
                    out.append(left + right)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols = out
        return [self._lookup[s] for s in symbols if s in self._lookup]

    def decode_token(self, token_id: int) -> Symbol:
        return self.tokens[token_id]

    # ------------------------------------------------------------------ io

    def save(self, path) -> None:
        with open(path, "w") as fh:
            json.dump(
                {
                    "merges": [[list(a), list(b)] for a, b in self.merges],
                    "tokens": [list(t) for t in self.tokens],
                    "min_frequency": self.min_frequency,
                    "max_token_length": self.max_token_length,
                },
                fh,
            )

    @classmethod
    def load(cls, path) -> "BPEVocab":
        with open(path) as fh:
            raw = json.load(fh)
        vocab = cls(
            merges=[(tuple(a), tuple(b)) for a, b in raw["merges"]],
            tokens=[tuple(t) for t in raw["tokens"]],
            min_frequency=raw["min_frequency"],
            max_token_length=raw["max_token_length"],
        )
        return vocab


def _pair_counts(corpus: list[list[Symbol]]) -> Counter:
    counts: Counter = Counter()
    for span in corpus:
        for a, b in zip(span, span[1:]):
            # Guard 1: a self-pair is never a merge candidate.
            if a == b:
                continue
            counts[(a, b)] += 1
    return counts


def _apply_merge(corpus: list[list[Symbol]], left: Symbol, right: Symbol) -> None:
    for s, span in enumerate(corpus):
        i = 0
        out: list[Symbol] = []
        while i < len(span):
            if i + 1 < len(span) and span[i] == left and span[i + 1] == right:
                out.append(left + right)
                i += 2
            else:
                out.append(span[i])
                i += 1
        corpus[s] = out


def train(
    spans: Iterable[Sequence[int]],
    vocab_size: int = 200,
    min_frequency: int = 10,
    max_token_length: int = 20,
    verbose: bool = False,
) -> BPEVocab:
    """Learn merges from a corpus of event spans.

    Args:
        spans: one sequence of base codes per completed event. Boundaries are
            implicit in the split, which is guard 2.
        vocab_size: target token count including the base alphabet.
    """
    corpus: list[list[Symbol]] = [[(int(c),) for c in span] for span in spans]

    alphabet = sorted({s for span in corpus for s in span})
    tokens: list[Symbol] = list(alphabet)
    merges: list[tuple[Symbol, Symbol]] = []

    while len(tokens) < vocab_size:
        counts = _pair_counts(corpus)
        if not counts:
            break
        (left, right), freq = counts.most_common(1)[0]
        if freq < min_frequency:
            break
        if len(left) + len(right) > max_token_length:
            # Skip this pair permanently by merging the next best instead.
            candidates = [
                (p, f)
                for p, f in counts.most_common()
                if len(p[0]) + len(p[1]) <= max_token_length and f >= min_frequency
            ]
            if not candidates:
                break
            (left, right), freq = candidates[0]

        merged = left + right
        _apply_merge(corpus, left, right)
        merges.append((left, right))
        if merged not in tokens:
            tokens.append(merged)
        if verbose:
            print(f"  merge {left}+{right} -> {merged}  (freq {freq})", flush=True)

    vocab = BPEVocab(
        merges=merges,
        tokens=tokens,
        min_frequency=min_frequency,
        max_token_length=max_token_length,
    )
    vocab.rebuild_lookup()
    return vocab


def assert_no_self_merges(vocab: BPEVocab) -> None:
    """Fail loudly if any learned merge collapses a repetition.

    Cheap enough to call after every training run, and the failure it catches is
    the one that silently invalidates the paper.
    """
    for left, right in vocab.merges:
        if left == right:
            raise AssertionError(f"self-merge learned: {left} + {right}")
    for token in vocab.tokens:
        if len(token) >= 2 and len(set(token)) == 1:
            raise AssertionError(
                f"token {token} is a pure repetition of one code; counts would be lost"
            )
