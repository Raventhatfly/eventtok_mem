"""Append-only event log, built from a code stream that arrives one step at a time.

**Why this module exists.** Text BPE is applied to a complete sequence. A robot's
codes arrive sequentially, and applying BPE to the growing prefix is *not*
stable: if a merge ``(A,B) -> X`` exists, the stream ``A`` tokenizes to ``[A]``
and one step later ``A B`` tokenizes to ``[X]``. The earlier token did not get
appended to — it was replaced. A log built that way rewrites its own history, and
any count read from it can change retroactively. That would defeat the entire
point of the design.

**The fix is where BPE runs, not whether.** Merges are confined to a single
completed event span:

    codes:  A A A | B B | C C C C | B B ...
                  ^ boundary -> span [A A A] is final -> BPE it -> append

Once a span is closed, nothing after the boundary may merge into it, so its
tokens are immutable. The log is append-only **at event granularity**. Latency is
one transition: a span is known to be closed one step after it closes.

Two guards from the vocabulary side make this sound (see ``build_vocab``):
self-merges are forbidden, so repetitions can never collapse; and boundaries are
hard barriers during vocabulary training too, so no learned merge ever spans one.

**Online boundary detectors only.** ``CodeChangeBoundary`` below and the
gripper/velocity heuristic both work from the past alone. UVD does *not* — it
walks backward from the final frame, so it is an offline evaluation tool and must
never appear in a test-time path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence


class BoundaryDetector(Protocol):
    """Decides, online, whether the event in progress has just ended."""

    def __call__(self, code: int, prev_code: int | None, span_len: int) -> bool: ...


@dataclass
class CodeChangeBoundary:
    """Boundary when the code changes and the span has run long enough.

    ``min_span`` guards against single-transition spans from a jittery code,
    which would otherwise flood the log. LOVE needed the same floor (their
    ``T_min = 3``) for the same reason: degenerate one-step segments are a local
    optimum that is hard to escape.
    """

    min_span: int = 2

    def __call__(self, code: int, prev_code: int | None, span_len: int) -> bool:
        if prev_code is None:
            return False
        return code != prev_code and span_len >= self.min_span


@dataclass
class EventLog:
    """Immutable, append-only sequence of completed events."""

    tokens: list[int] = field(default_factory=list)
    spans: list[tuple[int, int]] = field(default_factory=list)   # (start, end) transition idx
    codes: list[list[int]] = field(default_factory=list)         # raw codes per event

    def __len__(self) -> int:
        return len(self.tokens)

    def count(self, token: int) -> int:
        """How many times an event token occurs. The whole point of the design."""
        return self.tokens.count(token)

    def counts(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for t in self.tokens:
            out[t] = out.get(t, 0) + 1
        return out

    def run_length_encode(self) -> list[tuple[int, int]]:
        """``[(token, run), ...]`` — lossless compression for the prompt.

        Collapses *adjacent* repetitions only, so the multiplicity is preserved
        exactly (``SCOOP x7`` rather than "scooping"). This is the compression
        axis that keeps counts, as distinct from merging distant occurrences.
        """
        out: list[tuple[int, int]] = []
        for t in self.tokens:
            if out and out[-1][0] == t:
                out[-1] = (t, out[-1][1] + 1)
            else:
                out.append((t, 1))
        return out


class StreamingTokenizer:
    """Feed codes in one at a time; get an append-only event log out.

    Args:
        encode_span: maps a completed span of codes to one or more event tokens.
            This is where the trained BPE merges are applied. Applying them per
            span rather than to the global prefix is what makes the log stable.
        boundary: online boundary detector.
    """

    def __init__(
        self,
        encode_span: Callable[[Sequence[int]], list[int]],
        boundary: BoundaryDetector | None = None,
    ) -> None:
        self.encode_span = encode_span
        self.boundary = boundary or CodeChangeBoundary()
        self.log = EventLog()
        self._span: list[int] = []
        self._span_start = 0
        self._t = 0
        self._prev: int | None = None

    def push(self, code: int) -> list[int]:
        """Consume one transition's code. Returns tokens appended this step (often none)."""
        appended: list[int] = []
        if self.boundary(code, self._prev, len(self._span)):
            appended = self._close_span()
        self._span.append(int(code))
        self._prev = int(code)
        self._t += 1
        return appended

    def _close_span(self) -> list[int]:
        if not self._span:
            return []
        tokens = self.encode_span(self._span)
        self.log.tokens.extend(tokens)
        self.log.spans.append((self._span_start, self._t))
        self.log.codes.append(list(self._span))
        self._span = []
        self._span_start = self._t
        return tokens

    def finish(self) -> list[int]:
        """Close the trailing span. Only valid at episode end.

        Mid-episode the in-progress event is deliberately absent from the log:
        the log carries the past, and the policy already observes the present.
        """
        return self._close_span()


def identity_span_encoder(span: Sequence[int]) -> list[int]:
    """Placeholder until the BPE vocabulary exists: one token per span.

    Uses the span's **mode**, not its first code. Taking ``span[0]`` lets a single
    stray transition at the head of a span decide the whole event's identity —
    ``min_span`` stops a blip from *closing* a span, but the blip still lands at
    the front of the next one. The mode is robust to that, and is also the right
    reading semantically: an event is the code it spends most of its time in.

    Ties resolve to the earliest code, so the mapping stays deterministic.
    """
    counts: dict[int, int] = {}
    for c in span:
        counts[int(c)] = counts.get(int(c), 0) + 1
    best = max(counts.items(), key=lambda kv: (kv[1], -list(counts).index(kv[0])))
    return [best[0]]
