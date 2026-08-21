"""Finite Scalar Quantization, ported from the Flax implementation in
``robomme_policy_learning/src/openpi/models/utils/fsq_tokenizer.py``.

FSQ projects to ``d`` dimensions (d < 10), squashes each with tanh, and rounds
each to one of ``L_i`` levels. The implied codebook is the product of the
per-channel level sets, so ``|C| = prod(L_i)`` with no learned codebook at all —
hence no commitment loss, no EMA, no k-means init, no dead-code replacement, and
~100% codebook utilisation by construction. Mentzer et al., ICLR 2024.

Why FSQ here rather than VQ: QueST runs FSQ ``[8,5,5,5]`` = 1000 over robot action
chunks and beats VQ 89.8 vs 81.2 on LIBERO-90. FSQ's own paper notes VQ edges it
below |C| = 2^10, but VQ at 1024 codes sat at 57.5% codebook usage in BSQ's
measurements and collapsed to 27% on a low-diversity dataset in the
rotation-trick paper. The research risk in this project belongs in the training
objective, not in babysitting a codebook.

The parameterisation follows the Flax original: tanh maps to [-1, 1], which is
then discretised onto a uniform ``L_i``-point grid over that interval. That is the
same partition as the paper's ``floor(L/2) * tanh`` formulation, differently
scaled.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# From the FSQ paper, matching _get_bins_fsq in the Flax original. The 2**9
# entry is added from the paper's appendix: 512 codes at d=3 is the smallest
# configuration that keeps every L_i >= 5, which is their stated heuristic.
_BINS_BY_SIZE = {
    2**8: (8, 6, 5),           # 240
    2**9: (8, 8, 8),           # 512
    2**10: (8, 5, 5, 5),       # 1000
    2**12: (7, 5, 5, 5, 5),    # 4375
    2**14: (8, 8, 8, 6, 5),
    2**16: (8, 8, 8, 5, 5, 5),
}


def bins_for_size(target: int) -> tuple[int, ...]:
    if target not in _BINS_BY_SIZE:
        raise ValueError(
            f"no level set for target codebook size {target}; "
            f"known: {sorted(_BINS_BY_SIZE)}. Pass explicit levels instead."
        )
    return _BINS_BY_SIZE[target]


class FSQ(nn.Module):
    """Quantize ``(..., input_dim)`` to integer token ids and back.

    Args:
        input_dim: width of the incoming features.
        levels: per-channel level counts, e.g. ``(8, 8, 8)``. The paper's
            heuristic is ``L_i >= 5``; smaller values are reported as "subpar".
        target_codebook_size: alternative to ``levels``, looked up in the paper's
            table.
    """

    def __init__(
        self,
        input_dim: int,
        levels: tuple[int, ...] | None = None,
        target_codebook_size: int | None = None,
    ) -> None:
        super().__init__()
        if (levels is None) == (target_codebook_size is None):
            raise ValueError("pass exactly one of levels or target_codebook_size")
        if levels is None:
            levels = bins_for_size(int(target_codebook_size))
        if any(l < 2 for l in levels):
            raise ValueError(f"levels must be >= 2, got {levels}")
        if any(l < 5 for l in levels):
            # Not fatal, but the paper is explicit that it degrades.
            import warnings

            warnings.warn(
                f"FSQ levels {levels} include L_i < 5; the paper reports subpar "
                f"performance below 5 levels per channel.",
                stacklevel=2,
            )

        self.levels = tuple(int(l) for l in levels)
        self.dim = len(self.levels)
        self.input_dim = input_dim

        self.proj_down = nn.Linear(input_dim, self.dim)
        self.proj_up = nn.Linear(self.dim, input_dim)

        bases = torch.tensor(self.levels, dtype=torch.long)
        # Mixed-radix place values, least-significant channel first.
        place = torch.ones(self.dim, dtype=torch.long)
        for i in range(1, self.dim):
            place[i] = place[i - 1] * bases[i - 1]
        self.register_buffer("bases", bases, persistent=False)
        self.register_buffer("place", place, persistent=False)

    @property
    def codebook_size(self) -> int:
        return math.prod(self.levels)

    # ------------------------------------------------------------------ pieces

    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``z`` in [-1, 1] -> (digits, dequantized z) with a straight-through grad."""
        bases = self.bases.to(z.device)
        scale = (bases - 1).to(z.dtype)
        digits = torch.round((z + 1.0) * scale / 2.0)
        digits = digits.clamp(min=torch.zeros_like(scale), max=scale)
        z_q = digits / scale * 2.0 - 1.0
        # Straight-through: forward uses z_q, gradient flows to z.
        z_q = z + (z_q - z).detach()
        return digits.to(torch.long), z_q

    def pack(self, digits: torch.Tensor) -> torch.Tensor:
        """Per-channel digits -> a single integer token id."""
        return (digits * self.place.to(digits.device)).sum(dim=-1)

    def unpack(self, tokens: torch.Tensor) -> torch.Tensor:
        """Token ids -> per-channel digits."""
        place = self.place.to(tokens.device)
        bases = self.bases.to(tokens.device)
        return torch.div(tokens.unsqueeze(-1), place, rounding_mode="floor") % bases

    # ------------------------------------------------------------------ api

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(..., input_dim)`` -> (token ids ``(...)``, dequantized ``(..., dim)``)."""
        z = torch.tanh(self.proj_down(x))
        digits, z_q = self.quantize(z)
        return self.pack(digits), z_q

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Token ids -> ``(..., input_dim)``. No gradient path; for inspection."""
        digits = self.unpack(tokens)
        scale = (self.bases.to(tokens.device) - 1).to(self.proj_up.weight.dtype)
        z_q = digits.to(scale.dtype) / scale * 2.0 - 1.0
        return self.proj_up(z_q)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(..., input_dim)`` -> (token ids, reconstructed ``(..., input_dim)``)."""
        tokens, z_q = self.encode(x)
        return tokens, self.proj_up(z_q)

    def level_histogram(self, digits: torch.Tensor) -> list[torch.Tensor]:
        """Per-channel level occupancy.

        Aggregate codebook usage is the wrong instrument for FSQ: it cannot have
        dead codes in the VQ sense, but a channel whose marginal collapses onto
        one or two levels silently costs a factor of ``L_i`` of effective
        vocabulary while total usage still looks healthy. With d = 3 that is a
        factor of 8. Log this, not just usage.
        """
        flat = digits.reshape(-1, self.dim)
        return [
            torch.bincount(flat[:, i], minlength=self.levels[i])
            for i in range(self.dim)
        ]
