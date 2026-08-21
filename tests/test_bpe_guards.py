"""The two BPE guards.

The self-merge guard is the highest-stakes assertion in this repo. If a repeated
event collapses into one token, every count reported in the paper is an artifact
while the pipeline still runs cleanly and produces plausible numbers. It is a
test rather than a comment for exactly that reason.
"""

from __future__ import annotations

import pytest

from eventtok.bpe.build_vocab import BPEVocab, assert_no_self_merges, train


def test_repeated_pair_is_never_merged() -> None:
    """`S S` is the most frequent pair in the corpus and must survive anyway.

    Plain BPE would merge it on the first iteration, since a repeated action is
    what these tasks consist of.
    """
    spans = [[7, 7, 7, 7]] * 100
    vocab = train(spans, vocab_size=50, min_frequency=2)
    assert vocab.merges == [], "a self-pair was merged"
    assert_no_self_merges(vocab)
    # And the span still tokenizes to four separate tokens.
    assert len(vocab.encode_span([7, 7, 7, 7])) == 4


def test_counts_survive_training_on_repetitive_data() -> None:
    spans = [[1, 2], [1, 2], [1, 2]] * 60 + [[3, 3, 3]] * 60
    vocab = train(spans, vocab_size=64, min_frequency=5)
    assert_no_self_merges(vocab)
    encoded = vocab.encode_span([3, 3, 3])
    assert len(encoded) == 3, "a run of one code collapsed"


def test_distinct_pair_does_get_merged() -> None:
    """The guard must not disable BPE entirely."""
    spans = [[1, 2, 5]] * 100
    vocab = train(spans, vocab_size=32, min_frequency=5)
    assert vocab.merges, "no merges learned on an obviously mergeable corpus"
    assert len(vocab.encode_span([1, 2, 5])) < 3


def test_no_merge_crosses_a_span_boundary() -> None:
    """`(9, 4)` is frequent only across spans, so it must never be learned."""
    spans = [[9], [4]] * 200
    vocab = train(spans, vocab_size=32, min_frequency=2)
    assert vocab.merges == []
    for token in vocab.tokens:
        assert len(token) == 1


def test_min_frequency_respected() -> None:
    spans = [[1, 2]] * 3 + [[5, 6]] * 50
    vocab = train(spans, vocab_size=32, min_frequency=10)
    learned = {(a, b) for a, b in vocab.merges}
    assert ((1,), (2,)) not in learned
    assert ((5,), (6,)) in learned


def test_max_token_length_respected() -> None:
    spans = [[1, 2, 3, 4, 5, 6]] * 200
    vocab = train(spans, vocab_size=64, min_frequency=5, max_token_length=3)
    for token in vocab.tokens:
        assert len(token) <= 3


def test_round_trip_save_load(tmp_path) -> None:
    spans = [[1, 2, 5]] * 100
    vocab = train(spans, vocab_size=32, min_frequency=5)
    path = tmp_path / "vocab.json"
    vocab.save(path)
    loaded = BPEVocab.load(path)
    assert loaded.merges == vocab.merges
    assert loaded.tokens == vocab.tokens
    assert loaded.encode_span([1, 2, 5]) == vocab.encode_span([1, 2, 5])


def test_assert_no_self_merges_catches_a_bad_vocab() -> None:
    bad = BPEVocab(merges=[((4,), (4,))], tokens=[(4,), (4, 4)])
    with pytest.raises(AssertionError):
        assert_no_self_merges(bad)


def test_encode_is_deterministic() -> None:
    spans = [[1, 2, 3]] * 100 + [[2, 3, 4]] * 100
    vocab = train(spans, vocab_size=48, min_frequency=5)
    a = vocab.encode_span([1, 2, 3, 2, 3, 4])
    b = vocab.encode_span([1, 2, 3, 2, 3, 4])
    assert a == b
