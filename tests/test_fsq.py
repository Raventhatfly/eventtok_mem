"""FSQ correctness. Cheap, no data required."""

from __future__ import annotations

import math

import pytest
import torch

from eventtok.models.fsq import FSQ, bins_for_size


def test_paper_bin_tables() -> None:
    assert bins_for_size(2**8) == (8, 6, 5)
    assert bins_for_size(2**9) == (8, 8, 8)
    assert bins_for_size(2**10) == (8, 5, 5, 5)
    assert math.prod(bins_for_size(2**10)) == 1000
    assert math.prod(bins_for_size(2**9)) == 512


def test_codebook_size() -> None:
    assert FSQ(16, levels=(8, 8, 8)).codebook_size == 512
    assert FSQ(16, levels=(8, 5, 5, 5)).codebook_size == 1000


def test_pack_unpack_round_trip() -> None:
    """Mixed-radix packing must be a bijection over the whole codebook."""
    q = FSQ(16, levels=(8, 6, 5))
    grids = torch.meshgrid(*[torch.arange(l) for l in q.levels], indexing="ij")
    digits = torch.stack([g.reshape(-1) for g in grids], dim=-1)
    tokens = q.pack(digits)
    assert tokens.min().item() == 0
    assert tokens.max().item() == q.codebook_size - 1
    assert len(torch.unique(tokens)) == q.codebook_size
    assert torch.equal(q.unpack(tokens), digits)


def test_quantize_is_on_the_grid() -> None:
    q = FSQ(16, levels=(8, 8, 8))
    z = torch.empty(4096, 3).uniform_(-1.0, 1.0)
    digits, z_q = q.quantize(z)
    assert digits.min() >= 0
    for i, l in enumerate(q.levels):
        assert digits[:, i].max() < l
    # Dequantized values must land exactly on the uniform grid over [-1, 1].
    scale = torch.tensor(q.levels, dtype=z.dtype) - 1
    expected = digits.to(z.dtype) / scale * 2.0 - 1.0
    assert torch.allclose(z_q, expected, atol=1e-6)


def test_quantize_clamps_out_of_range_input() -> None:
    """tanh keeps z in [-1,1], but the clamp must hold if that ever changes."""
    q = FSQ(16, levels=(8, 8, 8))
    digits, _ = q.quantize(torch.tensor([[-5.0, 5.0, 0.0]]))
    assert digits.tolist() == [[0, 7, 3]] or digits.tolist() == [[0, 7, 4]]


def test_straight_through_gradient() -> None:
    """Forward is quantized; the gradient must reach the pre-quantization tensor."""
    q = FSQ(16, levels=(8, 8, 8))
    z = torch.empty(32, 3).uniform_(-0.9, 0.9).requires_grad_(True)
    _, z_q = q.quantize(z)
    z_q.sum().backward()
    assert z.grad is not None
    # d(z_q)/d(z) is identity under the STE.
    assert torch.allclose(z.grad, torch.ones_like(z))


def test_forward_shapes_and_determinism() -> None:
    q = FSQ(64, levels=(8, 8, 8)).eval()
    x = torch.randn(8, 5, 64)
    with torch.no_grad():
        tokens_a, recon = q(x)
        tokens_b, _ = q(x)
    assert tokens_a.shape == (8, 5)
    assert recon.shape == (8, 5, 64)
    assert torch.equal(tokens_a, tokens_b)
    assert tokens_a.max() < q.codebook_size


def test_level_histogram_counts_every_sample() -> None:
    q = FSQ(64, levels=(8, 8, 8))
    with torch.no_grad():
        _, z_q = q.quantize(torch.empty(500, 3).uniform_(-1, 1))
        digits, _ = q.quantize(torch.empty(500, 3).uniform_(-1, 1))
    hist = q.level_histogram(digits)
    assert len(hist) == 3
    for i, h in enumerate(hist):
        assert h.sum().item() == 500
        assert h.numel() == q.levels[i]


def test_warns_below_five_levels() -> None:
    with pytest.warns(UserWarning, match="subpar"):
        FSQ(16, levels=(2, 2, 2))
