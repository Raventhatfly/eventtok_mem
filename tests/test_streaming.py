"""The append-only property, tested directly.

If any of these fail, the log rewrites its own history and every count derived
from it is unreliable — which is the one thing this design cannot survive.
"""

from __future__ import annotations

from eventtok.bpe.streaming import (
    CodeChangeBoundary,
    EventLog,
    StreamingTokenizer,
    identity_span_encoder,
)


def run(codes, **kw) -> StreamingTokenizer:
    st = StreamingTokenizer(identity_span_encoder, CodeChangeBoundary(**kw))
    for c in codes:
        st.push(c)
    st.finish()
    return st


def test_log_is_append_only_under_growth() -> None:
    """Feeding a longer prefix must never change an already-emitted token.

    This is the property naive BPE-on-the-growing-prefix violates.
    """
    codes = [1, 1, 2, 2, 3, 3, 3, 1, 1, 2, 2]
    history: list[list[int]] = []
    st = StreamingTokenizer(identity_span_encoder, CodeChangeBoundary(min_span=2))
    for c in codes:
        st.push(c)
        history.append(list(st.log.tokens))
    for earlier, later in zip(history, history[1:]):
        assert later[: len(earlier)] == earlier, (
            f"log was rewritten: {earlier} -> {later}"
        )


def test_repetitions_are_preserved() -> None:
    """Three separated occurrences of the same event stay three tokens."""
    st = run([7, 7, 9, 9, 7, 7, 9, 9, 7, 7, 9, 9], min_span=2)
    assert st.log.count(7) == 3
    assert st.log.count(9) == 3
    assert st.log.tokens == [7, 9, 7, 9, 7, 9]


def test_adjacent_repeats_do_not_split_into_many_events() -> None:
    """A long run of one code is one event, not one event per transition."""
    st = run([5] * 10 + [6] * 4, min_span=2)
    assert st.log.tokens == [5, 6]


def test_min_span_suppresses_single_step_jitter() -> None:
    """A one-transition blip must not open an event of its own."""
    st = run([1, 1, 1, 2, 1, 1, 1], min_span=3)
    # The lone 2 is absorbed rather than emitted as its own event.
    assert 2 not in st.log.tokens


def test_no_tokens_emitted_before_first_boundary() -> None:
    st = StreamingTokenizer(identity_span_encoder, CodeChangeBoundary(min_span=2))
    for c in [4, 4, 4]:
        st.push(c)
    assert st.log.tokens == []          # event still in progress
    st.finish()
    assert st.log.tokens == [4]


def test_run_length_encoding_is_lossless_for_counts() -> None:
    log = EventLog(tokens=[3, 3, 3, 3, 3, 3, 3, 8])
    assert log.run_length_encode() == [(3, 7), (8, 1)]
    # The multiplicity survives, unlike a merge that would render this "3, 8".
    assert sum(run for _, run in log.run_length_encode() if _ == 3) == log.count(3)


def test_counts_dict() -> None:
    st = run([1, 1, 2, 2, 1, 1, 3, 3, 1, 1], min_span=2)
    assert st.log.counts() == {1: 3, 2: 1, 3: 1}


def test_spans_tile_the_stream_without_overlap() -> None:
    st = run([1, 1, 2, 2, 2, 3, 3], min_span=2)
    spans = st.log.spans
    assert spans[0][0] == 0
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end == start, "spans must be contiguous and non-overlapping"
    assert spans[-1][1] == 7
