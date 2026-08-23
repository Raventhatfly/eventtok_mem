"""Does a policy actually use the event log? The control that can invalidate everything.

**Scope, stated first because it limits the conclusion.** There is no simulator on this
machine (no robosuite, no mujoco), so this is not rollout success. It is a
behaviour-cloning probe: predict the next action chunk from the current observation,
with and without the memory log, and see whether the log helps and whether the policy
cares what it says. A log that cannot improve action prediction offline will not
improve a rollout; the converse is not guaranteed, so a positive result here is
necessary and not sufficient.

Five conditions, and the contrasts between them are the point:

    none      observation only. The floor.
    log       the correct event-token log, as a *sequence*.
    wrong     a log from a different episode of the same task, length-matched.
    shuffled  the correct tokens in a random order -- same multiset, order destroyed.
    count     a count vector over the vocabulary. No order, no sequence, just how many
              times each token has occurred: the hand-designed "derived state" the plan
              names as a baseline that may well win.

    log vs none      does memory help at all
    log vs wrong     does the *content* matter, or merely having something there
    log vs shuffled  does order matter, or only the multiset
    log vs count     does the raw log beat simply counting

**log vs wrong is the one that can end the project.** If a deliberately incorrect
memory scores like a correct one, the policy is ignoring the memory and every transfer
number computed on top of it means nothing.

**No future leakage.** The log at transition t contains only tokens whose span has
*closed* by t. Conditioning on the whole-episode log would tell the policy at the first
repetition that there will be three, which is the silent failure that would inflate
exactly the counting result this project reports.

The memory encoder is order-sensitive on purpose -- token embedding plus a positional
embedding through a small transformer. Summing embeddings would preserve counts but
make ``shuffled`` identical to ``log`` by construction and the contrast vacuous.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

CONDITIONS = ("none", "log", "wrong", "shuffled", "count")


class MemoryPolicy(nn.Module):
    def __init__(
        self,
        vocab: int,
        d_feat: int = 2048,
        n_vis: int = 4,
        state_dim: int = 8,
        action_dim: int = 8,
        k: int = 20,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        max_log: int = 64,
        condition: str = "log",
    ) -> None:
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}")
        self.condition = condition
        self.k = k
        self.action_dim = action_dim
        self.max_log = max_log
        self.vocab = vocab

        self.vis_in = nn.Sequential(nn.LayerNorm(d_feat), nn.Linear(d_feat, d_model))
        self.state_in = nn.Linear(state_dim, d_model)

        mem_dim = 0
        if condition in ("log", "wrong", "shuffled"):
            self.tok_emb = nn.Embedding(vocab + 1, d_model)      # +1 = padding
            self.pos_emb = nn.Embedding(max_log, d_model)
            self.empty = nn.Parameter(torch.zeros(d_model))
            layer = nn.TransformerEncoderLayer(
                d_model, n_heads, d_model * 4, batch_first=True, norm_first=True
            )
            self.mem_enc = nn.TransformerEncoder(layer, n_layers)
            mem_dim = d_model
        elif condition == "count":
            self.count_in = nn.Sequential(
                nn.Linear(vocab, d_model), nn.GELU(), nn.Linear(d_model, d_model)
            )
            mem_dim = d_model

        self.head = nn.Sequential(
            nn.Linear(d_model * (1 + n_vis) + mem_dim, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, k * action_dim),
        )

    def encode_memory(self, log_tokens: torch.Tensor, log_len: torch.Tensor):
        """``(B, max_log)`` padded token ids -> ``(B, d_model)``."""
        b, L = log_tokens.shape
        pos = torch.arange(L, device=log_tokens.device).unsqueeze(0).expand(b, L)
        x = self.tok_emb(log_tokens) + self.pos_emb(pos)
        pad = torch.arange(L, device=log_tokens.device).unsqueeze(0) >= log_len.unsqueeze(1)
        x = self.mem_enc(x, src_key_padding_mask=pad)
        keep = (~pad).float().unsqueeze(-1)
        pooled = (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        # An empty log is a real state -- nothing has happened yet -- and must not be
        # silently read as a zero vector that also means "no memory available".
        return torch.where((log_len == 0).unsqueeze(1), self.empty.expand(b, -1), pooled)

    def forward(self, feat: torch.Tensor, state: torch.Tensor,
                log_tokens=None, log_len=None, counts=None) -> torch.Tensor:
        parts = [self.vis_in(feat).flatten(1), self.state_in(state)]
        if self.condition in ("log", "wrong", "shuffled"):
            parts.append(self.encode_memory(log_tokens, log_len))
        elif self.condition == "count":
            parts.append(self.count_in(counts))
        h = torch.cat(parts, dim=-1)
        return self.head(h).view(-1, self.k, self.action_dim)


def build_logs(streams, vocab, min_span: int = 3):
    """``{epis_idx: [(token, start, end), ...]}`` -- the log with spans."""
    from ..eval.bpe_boundaries import encode_aligned, runs_with_spans

    return {
        idx: encode_aligned(vocab, runs_with_spans(codes, min_span))
        for idx, codes in streams.items()
    }


def prefix_tokens(log, t: int, max_log: int) -> list[int]:
    """Tokens closed strictly before transition ``t``. The no-leakage rule."""
    out = [tok for tok, _, hi in log if hi <= t]
    return out[-max_log:]
