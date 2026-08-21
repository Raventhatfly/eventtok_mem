"""Event tokenizer: a learned quantizer over action chunks, optionally
conditioned on vision.

    encoder(action chunk [, visual context])  ->  registers  ->  FSQ  ->  code
    head(code)                                ->  action chunk        (L1)
    far_head(code, feat_t)                    ->  feat_{t+far}        (cosine, optional)

**This is deliberately an autoencoder, not a prediction model**, and that is the
point. Plain k-means over normalised delta-action chunks already produces the
property the design needs, measured on 40 SwingXtimes episodes:

    K=32 clusters:  within-event code change 11.6%,  MI with the event label
                    59.5% of label entropy,  a recurring n-gram matching the
                    target repetition count N in 29/40 episodes

and critically, **repeated instances of an event land on the same cluster** —
both right-side visits map to one symbol, both left-side visits to another. That
invariance is the hard part, and clustering the action chunks gives it directly.
This module is the learnable version of that (which is what QueST and OAT are).

**What earlier versions got wrong, in order.** Feeding actions in *and* predicting
them alongside a ``feat_next`` residual made every target present in the input, so
the only pressure was the FSQ bottleneck and the code became a smooth positional
index (consecutive codes differing by the FSQ place values 1/8/64). Removing the
leak by going vision-only then collapsed entirely — 1/64 codes — because inferring
the action from 2x2 visual features is far harder than clustering the action
itself. The resolution is that **event identity lives in the action**; vision
belongs on the target side, where it can supply outcome ("did the grasp succeed"),
not on the input side, where it pulls the code toward trajectory phase.

**A caveat that must not be dropped.** The code changes ~4-8x more often than
events do (boundary precision 0.13-0.24 at recall ~1.0). It is a high-recall
*candidate* symbol stream, not a segmentation. Turning candidates into events is
the BPE stage's job — merging frequent adjacent pairs is exactly how 7x
over-segmentation collapses into event-sized units. Do not call a code change an
event boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .encoder import RegisterEncoder
from .fsq import FSQ

# Modality ids for the encoder's token-type embedding.
M_ACTION, M_FEAT_T, M_DELTA = 0, 1, 2


@dataclass
class TokenizerOutput:
    tokens: torch.Tensor                  # (B, R) integer code ids
    z_q: torch.Tensor                     # (B, R, d_model)
    action_pred: torch.Tensor             # (B, k, action_dim)
    digits: torch.Tensor                  # (B, R, fsq_dim)
    far_pred: torch.Tensor | None = None  # (B, n_vis, d_feat)


class EventTokenizer(nn.Module):
    """Args:
        use_vision: add ``feat_t`` and the visual delta as encoder context. Off by
            default — vision on the input side measurably pulled the code toward
            trajectory phase rather than motion type.
        far_head: predict a frame beyond the input horizon, for outcome signal.
    """

    def __init__(
        self,
        action_dim: int = 8,
        k: int = 20,
        d_feat: int = 2048,
        n_vis_tokens: int = 4,
        d_model: int = 256,
        n_registers: int = 2,
        n_layers: int = 3,
        n_heads: int = 4,
        fsq_levels: tuple[int, ...] = (8, 8),
        causal_registers: bool = True,
        use_vision: bool = False,
        far_head: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.k = k
        self.d_feat = d_feat
        self.n_vis_tokens = n_vis_tokens
        self.n_registers = n_registers
        self.use_vision = use_vision

        # Each action step is one token, so the encoder sees the chunk's temporal
        # structure rather than a flattened vector.
        self.action_in = nn.Linear(action_dim, d_model)
        self.vision_in = nn.Linear(d_feat, d_model) if use_vision else None

        self.encoder = RegisterEncoder(
            d_in=d_model,
            d_model=d_model,
            n_registers=n_registers,
            n_layers=n_layers,
            n_heads=n_heads,
            causal_registers=causal_registers,
            n_modalities=3,
        )
        self.fsq = FSQ(input_dim=d_model, levels=fsq_levels)

        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, k * action_dim),
        )

        self.far_head = None
        if far_head:
            self.feat_ctx = nn.Sequential(
                nn.LayerNorm(d_feat), nn.Linear(d_feat, d_model)
            )
            self.far_head = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_feat),
            )

    @property
    def codebook_size(self) -> int:
        return self.fsq.codebook_size

    def encode(
        self,
        actions: torch.Tensor,
        feat_t: torch.Tensor | None = None,
        feat_next: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parts = [self.action_in(actions)]
        ids = [torch.full((actions.shape[1],), M_ACTION, dtype=torch.long)]

        if self.use_vision and feat_t is not None and feat_next is not None:
            parts.append(self.vision_in(feat_t))
            ids.append(torch.full((feat_t.shape[1],), M_FEAT_T, dtype=torch.long))
            parts.append(self.vision_in(feat_next - feat_t))
            ids.append(torch.full((feat_t.shape[1],), M_DELTA, dtype=torch.long))

        tokens_in = torch.cat(parts, dim=1)
        modality_ids = torch.cat(ids).to(actions.device)

        registers = self.encoder(tokens_in, modality_ids)
        z = torch.tanh(self.fsq.proj_down(registers))
        digits, z_q_small = self.fsq.quantize(z)
        return self.fsq.pack(digits), self.fsq.proj_up(z_q_small), digits

    def forward(
        self,
        actions: torch.Tensor,
        feat_t: torch.Tensor | None = None,
        feat_next: torch.Tensor | None = None,
        feat_far: torch.Tensor | None = None,
    ) -> TokenizerOutput:
        tokens, z_q, digits = self.encode(actions, feat_t, feat_next)
        # The register axis is the coarse-to-fine ordering, not a temporal axis,
        # so pooling here does not touch count information.
        code = z_q.mean(dim=1)

        action_pred = self.action_head(code).view(-1, self.k, self.action_dim)

        far_pred = None
        if self.far_head is not None and feat_far is not None and feat_t is not None:
            ctx = self.feat_ctx(feat_t)
            code_b = code.unsqueeze(1).expand(-1, ctx.shape[1], -1)
            far_pred = self.far_head(torch.cat([ctx, code_b], dim=-1))

        return TokenizerOutput(tokens, z_q, action_pred, digits, far_pred)


def losses(
    out: TokenizerOutput,
    actions: torch.Tensor,
    feat_t: torch.Tensor | None = None,
    feat_far: torch.Tensor | None = None,
    w_action: float = 1.0,
    w_far: float = 0.0,
) -> dict[str, torch.Tensor]:
    """L1 on the action chunk; optional cosine on the far-future residual.

    Actions must already be per-dimension normalised — unnormalised, the gripper
    (std ~0.95) swamps the seven pose dimensions (std 0.05-0.20).

    Reference points on normalised SwingXtimes actions: predicting the dataset
    mean gives L1 0.6567, and a linear map from the visual delta gives 0.2371. A
    quantizer over the actions themselves should go well below both; if it does
    not, the bottleneck is too tight rather than the objective being wrong.
    """
    action_loss = F.l1_loss(out.action_pred, actions)
    result = {"action": action_loss, "total": w_action * action_loss}

    if out.far_pred is not None and feat_far is not None and feat_t is not None:
        target = feat_far - feat_t
        cos = F.cosine_similarity(out.far_pred.flatten(1), target.flatten(1), dim=-1)
        result["far"] = 1.0 - cos.mean()
        result["far_cos"] = cos.mean().detach()
        result["total"] = result["total"] + w_far * result["far"]
    return result
