"""Nested dropout: the ordering mechanism OAT credits with most of its benefit.

Registers are only a *set* of latents unless something forces them into an order. Nested
dropout supplies that: during training a keep-length ``m`` is sampled and registers
``m..R-1`` are zeroed, so the decoder must reconstruct from the first ``m`` alone. Under
that pressure register 0 has to carry the coarsest description, register 1 refines it,
and so on. At inference nothing is dropped, but the ordering persists -- which is what
makes a prefix of the registers a usable coarse code.

Why this module exists at all: the first pass at this project built the FSQ codebook and
the register encoder, concluded "the learned tokenizer does not beat k-means", and had
never implemented nested dropout -- the component OAT reports as worth 21 points on
LIBERO on its own. So the comparison was FSQ+registers against Lloyd's algorithm, not OAT
against Lloyd's algorithm, and the conclusion was stated more broadly than the evidence
supported.

The keep-length distribution is geometric over powers of two, following the original
nested-dropout formulation and OAT's use of it: short prefixes are sampled far more often
than long ones, because the coarse registers need the most gradient signal. With R=2 the
support is {1, 2}; with R=8 it is {1, 2, 4, 8}.
"""

from __future__ import annotations

import torch


def pow2_keep_lengths(n_registers: int) -> list[int]:
    """``[1, 2, 4, ..., n_registers]`` -- the prefix lengths that are ever kept.

    Restricting to powers of two rather than every length in ``1..R`` is deliberate: it
    concentrates the sampling on a few well-trained truncation points instead of spreading
    it thinly, and it matches how a coarse-to-fine code is actually consumed (take the
    first 1, 2, 4 ... registers).
    """
    lengths, m = [], 1
    while m < n_registers:
        lengths.append(m)
        m *= 2
    lengths.append(n_registers)
    return lengths


def sample_keep_length(
    n_registers: int, batch: int, device, generator: torch.Generator | None = None
) -> torch.Tensor:
    """One keep-length per batch element, geometric over the pow2 lengths.

    Weight ``2^-i`` on the i-th length, so length 1 is sampled about twice as often as
    length 2, four times as often as length 4. The coarse registers are used by every
    truncation, so they need the most signal; the finest one is used by exactly one.
    """
    lengths = pow2_keep_lengths(n_registers)
    w = torch.tensor([2.0 ** -i for i in range(len(lengths))], device=device)
    idx = torch.multinomial(w.expand(batch, -1), 1, replacement=True,
                            generator=generator).squeeze(1)
    return torch.tensor(lengths, device=device)[idx]


def apply_nested_dropout(
    z: torch.Tensor, keep: torch.Tensor | None = None, generator=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero registers past ``keep`` for each batch element.

    Args:
        z: ``(B, R, d)`` register latents.
        keep: ``(B,)`` keep-lengths, sampled if omitted.

    Returns ``(masked z, keep)``. Zeroing rather than slicing keeps the tensor shape
    fixed, so the same decoder handles every truncation and the batch stays rectangular.
    """
    b, r, _ = z.shape
    if keep is None:
        keep = sample_keep_length(r, b, z.device, generator)
    pos = torch.arange(r, device=z.device).unsqueeze(0)
    mask = (pos < keep.unsqueeze(1)).unsqueeze(-1).to(z.dtype)
    return z * mask, keep
