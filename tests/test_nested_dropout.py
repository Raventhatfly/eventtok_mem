"""Nested dropout must order the registers, and must not leak into inference.

This is the component OAT credits with 21 points on LIBERO, and the first version of this
project concluded "the learned tokenizer does not beat k-means" without having
implemented it. These tests pin the two properties that make it meaningful.
"""

from __future__ import annotations

import torch

from eventtok.models.nested_dropout import (
    apply_nested_dropout,
    pow2_keep_lengths,
    sample_keep_length,
)
from eventtok.models.tokenizer import EventTokenizer


def test_keep_lengths_are_powers_of_two_up_to_r() -> None:
    assert pow2_keep_lengths(1) == [1]
    assert pow2_keep_lengths(2) == [1, 2]
    assert pow2_keep_lengths(8) == [1, 2, 4, 8]
    assert pow2_keep_lengths(6) == [1, 2, 4, 6]      # non-power-of-two R still terminates


def test_suffix_is_zeroed_and_prefix_is_untouched() -> None:
    z = torch.randn(32, 8, 5)
    zm, keep = apply_nested_dropout(z)
    for i in range(len(z)):
        k = int(keep[i])
        assert torch.equal(zm[i, :k], z[i, :k]), "prefix must pass through unchanged"
        assert zm[i, k:].abs().sum() == 0, "suffix must be zeroed"


def test_short_prefixes_are_sampled_more_often() -> None:
    """Coarse registers are used by every truncation, so they need the most signal."""
    s = sample_keep_length(8, 40000, "cpu")
    freq = {k: (s == k).float().mean().item() for k in (1, 2, 4, 8)}
    assert freq[1] > freq[2] > freq[4] > freq[8]
    assert abs(freq[1] - 8 / 15) < 0.03      # geometric 2^-i over four lengths


def test_dropout_is_training_only() -> None:
    """At inference nothing may be dropped -- the ordering persists, the masking does not.

    If this leaked into eval, every reported code would be a random truncation and the
    tokenizer would look far worse than it is.
    """
    torch.manual_seed(0)
    m = EventTokenizer(n_registers=8, fsq_levels=(8, 8), nested_dropout=True)
    a = torch.randn(4, 20, 8)
    m.eval()
    with torch.no_grad():
        one = m(a).tokens
        two = m(a).tokens
    assert torch.equal(one, two), "eval must be deterministic"


def test_flag_defaults_off() -> None:
    """Off by default so the ablation is explicit rather than accidental."""
    assert EventTokenizer().nested_dropout is False
