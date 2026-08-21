"""The event tokenizer: transition in, discrete code out.

One transition is ``(feat_t, feat_{t+k}, actions)``. The encoder's registers are
quantized by FSQ, and two decoder heads read the code back out:

    head A   (feat_t, code) -> feat_{t+k}       cosine loss
    head B   (code)         -> action chunk     L1 loss

**The dual head is what forces both modalities into the code.** A code carrying
only the action fails head A, because the same action has different outcomes — a
gripper close that lifts a cube and one that closes on nothing are the same
command and different events, and that distinction is exactly what memory needs.
A code carrying only the visual change fails head B.

Head A is given ``feat_t`` for free, so the code only has to encode *what
changed*. That is why Genie's 8 codes come out meaning left/right/jump rather
than describing the scene.

Never reconstruct pixels. Five independent results say reconstruction-trained
codes are not event-aligned: FSQ's own appendix ("no evidence that a particular
code represents a fixed visual concept"), BEiT-v2's Table 4 where better
reconstruction costs 15 points of linear-probe accuracy, LOVE's identical-ELBO /
0.91-vs-0.82-boundary-F1 result, UniVLA's 88.7 vs 82.3, and REPA's analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .encoder import RegisterEncoder
from .fsq import FSQ

# Modality ids for the encoder's token-type embedding.
M_FEAT_T, M_FEAT_NEXT, M_ACTION = 0, 1, 2


@dataclass
class TokenizerOutput:
    tokens: torch.Tensor          # (B, R) integer code ids
    z_q: torch.Tensor             # (B, R, d_model) quantized registers
    feat_pred: torch.Tensor       # (B, n_vis, d_feat)
    action_pred: torch.Tensor     # (B, k, action_dim)
    digits: torch.Tensor          # (B, R, fsq_dim) per-channel levels


class EventTokenizer(nn.Module):
    def __init__(
        self,
        d_feat: int = 2048,
        n_vis_tokens: int = 4,
        action_dim: int = 8,
        k: int = 20,
        d_model: int = 256,
        n_registers: int = 2,
        n_layers: int = 3,
        n_heads: int = 4,
        fsq_levels: tuple[int, ...] = (8, 8, 8),
        causal_registers: bool = True,
    ) -> None:
        super().__init__()
        self.d_feat = d_feat
        self.n_vis_tokens = n_vis_tokens
        self.action_dim = action_dim
        self.k = k
        self.n_registers = n_registers

        # Actions are projected to the feature width so one encoder can attend
        # over a single concatenated token sequence.
        self.action_proj = nn.Linear(action_dim, d_feat)

        self.encoder = RegisterEncoder(
            d_in=d_feat,
            d_model=d_model,
            n_registers=n_registers,
            n_layers=n_layers,
            n_heads=n_heads,
            causal_registers=causal_registers,
        )
        self.fsq = FSQ(input_dim=d_model, levels=fsq_levels)

        # Head A: predict the feature residual, per visual token.
        #
        # Pooling feat_t to a single d_model vector and emitting a flat
        # n_vis*d_feat prediction was measurably worse than predicting the
        # dataset-mean residual (+0.0425 vs +0.0440 cosine), while a plain linear
        # map from the full feat_t reaches +0.6995. The bottleneck was the
        # context path, not the task. Keeping it per-token preserves that
        # information and keeps the head's output width at d_feat.
        self.feat_ctx = nn.Sequential(
            nn.LayerNorm(d_feat), nn.Linear(d_feat, d_model)
        )
        self.feat_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_feat),
        )
        # Head B: reconstruct the action chunk from the code alone.
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, k * action_dim),
        )

    @property
    def codebook_size(self) -> int:
        return self.fsq.codebook_size

    # ------------------------------------------------------------------ forward

    def encode(
        self, feat_t: torch.Tensor, feat_next: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (tokens ``(B, R)``, z_q ``(B, R, d_model)``, digits ``(B, R, fsq_dim)``)."""
        act_tokens = self.action_proj(actions)                    # (B, k, d_feat)
        tokens_in = torch.cat([feat_t, feat_next, act_tokens], dim=1)

        n_v = feat_t.shape[1]
        modality_ids = torch.cat(
            [
                torch.full((n_v,), M_FEAT_T, dtype=torch.long),
                torch.full((feat_next.shape[1],), M_FEAT_NEXT, dtype=torch.long),
                torch.full((act_tokens.shape[1],), M_ACTION, dtype=torch.long),
            ]
        ).to(feat_t.device)

        registers = self.encoder(tokens_in, modality_ids)         # (B, R, d_model)
        z = torch.tanh(self.fsq.proj_down(registers))
        digits, z_q_small = self.fsq.quantize(z)
        z_q = self.fsq.proj_up(z_q_small)                          # (B, R, d_model)
        return self.fsq.pack(digits), z_q, digits

    def forward(
        self, feat_t: torch.Tensor, feat_next: torch.Tensor, actions: torch.Tensor
    ) -> TokenizerOutput:
        tokens, z_q, digits = self.encode(feat_t, feat_next, actions)

        # Registers are pooled for the heads; the *sequence* of registers is the
        # coarse-to-fine axis, not a temporal axis, so mean-pooling here does not
        # touch the count information (which lives across transitions, not within).
        code = z_q.mean(dim=1)                                     # (B, d_model)

        # Head A predicts the *residual* feat_next - feat_t, not feat_next.
        # Measured on SwingXtimes, cos(feat_t, feat_next) at k=20 is 0.9984: one
        # second of robot motion barely moves the SigLIP features. Predicting the
        # absolute future therefore has a trivial solution — copy the input and
        # ignore the code — which drives the codebook to full collapse (all three
        # channels pinned to one level, 1/512 codes used). Predicting the residual
        # removes that shortcut, so the code has to carry the change.
        ctx = self.feat_ctx(feat_t)                                # (B, n_vis, d_model)
        code_b = code.unsqueeze(1).expand(-1, ctx.shape[1], -1)    # (B, n_vis, d_model)
        feat_pred = self.feat_head(torch.cat([ctx, code_b], dim=-1))  # (B, n_vis, d_feat)

        action_pred = self.action_head(code).view(-1, self.k, self.action_dim)

        return TokenizerOutput(tokens, z_q, feat_pred, action_pred, digits)


def losses(
    out: TokenizerOutput,
    feat_t: torch.Tensor,
    feat_next: torch.Tensor,
    actions: torch.Tensor,
    w_feat: float = 1.0,
    w_action: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Cosine on the *feature residual*, L1 on the action chunk.

    Cosine rather than MSE: the SigLIP features are unnormalised with absmax
    ~130 while the residual has std ~0.23, so an MSE on either would be
    dominated by magnitude and would spend the code on brightness rather than on
    what happened.

    The residual target is what stops the trivial copy solution — see the note in
    ``EventTokenizer.forward``.
    """
    target = feat_next - feat_t
    cos = F.cosine_similarity(out.feat_pred.flatten(1), target.flatten(1), dim=-1)
    feat_loss = 1.0 - cos.mean()
    action_loss = F.l1_loss(out.action_pred, actions)
    return {
        "feat": feat_loss,
        "action": action_loss,
        # Alignment with the true residual. 0 means no better than chance, so
        # this is the number that says whether the code carries anything.
        "residual_cos": cos.mean().detach(),
        "total": w_feat * feat_loss + w_action * action_loss,
    }
