"""Register-token encoder, ported from ``TokenizerEncoderDecoder`` (Flax).

A small set of learnable *register* queries cross-attends into the transition's
data tokens and is the only thing that survives to the quantizer. Registers are
the bottleneck, so the number of them is the token budget per transition.

Optional causal masking among registers, plus tail masking (nested dropout), give
the OAT ordering property: earlier registers carry coarse structure and later ones
refine it. OAT reports 21 points on LIBERO from nested dropout alone, so it is
worth having — but it is an M3 ablation here, not part of the first result.

Input LayerNorm matters: the SigLIP features arrive unnormalised with absmax
around 130, and feeding those straight into attention destroys the first few
hundred steps of training.
"""

from __future__ import annotations

import torch
from torch import nn


class CrossAttentionBlock(nn.Module):
    """Self-attention over registers, then cross-attention into the data."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_q1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_q2 = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        hidden = int(d_model * mlp_ratio)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        self_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm_q1(q)
        attn, _ = self.self_attn(h, h, h, attn_mask=self_mask, need_weights=False)
        q = q + attn

        h = self.norm_q2(q)
        attn, _ = self.cross_attn(h, self.norm_kv(kv), self.norm_kv(kv), need_weights=False)
        q = q + attn

        return q + self.mlp(self.norm_mlp(q))


class RegisterEncoder(nn.Module):
    """Data tokens -> ``n_registers`` latent vectors.

    Args:
        d_in: width of each incoming data token.
        d_model: internal width.
        n_registers: token budget per transition. Small on purpose — every
            latent-action model with semantically clean codes uses a tiny budget
            (Genie 8 codes, IGOR 32, UniVLA 16x2, Moto 128), not a large flat one.
        causal_registers: mask register self-attention so register ``i`` cannot
            see ``j > i``. Required for the ordering property to hold rather than
            merely be incentivised.
    """

    def __init__(
        self,
        d_in: int,
        d_model: int = 256,
        n_registers: int = 2,
        n_layers: int = 3,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        causal_registers: bool = True,
        n_modalities: int = 3,
    ) -> None:
        super().__init__()
        self.n_registers = n_registers
        self.d_model = d_model
        self.causal_registers = causal_registers

        self.in_norm = nn.LayerNorm(d_in)
        self.in_proj = nn.Linear(d_in, d_model)
        # Distinguishes feat_t / feat_next / action tokens, so the encoder can
        # tell "before" from "after" rather than inferring it from content.
        self.modality_embed = nn.Parameter(torch.randn(n_modalities, d_model) * 0.02)
        self.registers = nn.Parameter(torch.randn(n_registers, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(d_model, n_heads, mlp_ratio, dropout)
                for _ in range(n_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(d_model)

    def _register_mask(self, device) -> torch.Tensor | None:
        if not self.causal_registers or self.n_registers == 1:
            return None
        n = self.n_registers
        return torch.triu(
            torch.full((n, n), float("-inf"), device=device), diagonal=1
        )

    def forward(
        self,
        tokens: torch.Tensor,
        modality_ids: torch.Tensor,
    ) -> torch.Tensor:
        """``tokens (B, T, d_in)``, ``modality_ids (T,)`` -> ``(B, n_registers, d_model)``."""
        kv = self.in_proj(self.in_norm(tokens))
        kv = kv + self.modality_embed[modality_ids].unsqueeze(0)

        q = self.registers.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        mask = self._register_mask(tokens.device)
        for block in self.blocks:
            q = block(q, kv, self_mask=mask)
        return self.out_norm(q)
