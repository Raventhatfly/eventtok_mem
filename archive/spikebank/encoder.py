"""Novelty-gated sparse spike encoder: dense stream -> sparse binary code.

Design ref: docs/DESIGN.md section 4. Two jobs:
  1. gate write bandwidth by predictive surprise, so idle segments do not
     saturate the long-tau banks;
  2. lift z_t into a high-dimensional sparse code (FlyHash / SDM style) so that
     STDP updates on different observations do not interfere.
"""
from __future__ import annotations

import torch


class NoveltySparseEncoder(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        n_out: int = 2048,
        k: int = 64,
        proj_density: float = 0.1,
        lam: float = 0.3,
        gate: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.k, self.lam, self.gate = k, lam, gate
        w = (torch.rand(d_in, n_out, generator=g) < proj_density).float()
        w *= torch.randn(d_in, n_out, generator=g)
        self.register_buffer("w_enc", w)
        self.register_buffer("nov_mu", torch.zeros(1))
        self.register_buffer("nov_var", torch.ones(1))

    def init_state(self, batch: int, d_in: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, d_in, device=device or self.w_enc.device)

    @torch.no_grad()
    def forward(self, z_t: torch.Tensor, z_hat: torch.Tensor):
        """z_t: (B, d_in) dense obs embedding. z_hat: running prediction.

        Returns (x_t sparse binary code, z_hat_next, novelty gate g_t).
        """
        e_t = z_t - z_hat
        nov = e_t.norm(dim=-1, keepdim=True)
        if self.gate:
            self.nov_mu.mul_(0.99).add_(0.01 * nov.mean())
            self.nov_var.mul_(0.99).add_(0.01 * nov.var(unbiased=False).clamp_min(1e-8))
            g_t = torch.sigmoid((nov - self.nov_mu) / self.nov_var.sqrt().clamp_min(1e-6))
        else:
            g_t = torch.ones_like(nov)

        u = (e_t * g_t) @ self.w_enc
        idx = u.topk(self.k, dim=1).indices              # k-WTA -> exactly k spikes
        x_t = torch.zeros_like(u).scatter_(1, idx, 1.0)
        z_hat_next = (1 - self.lam) * z_hat + self.lam * z_t
        return x_t, z_hat_next, g_t
