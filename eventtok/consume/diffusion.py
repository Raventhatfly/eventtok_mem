"""A conditional Diffusion Policy over action chunks, with the event log as memory.

Chi et al.'s formulation, transformer variant: a denoiser over an action chunk,
trained to predict the noise added to it, conditioned on the current observation.
Written here rather than pulled from the upstream repo because that repo drags in
robomimic, robosuite, hydra and zarr, and this project has already been bitten once by
a dependency quietly changing the numpy pin. Everything below is torch only.

**How the memory enters, and why it is not a pooled global condition.** Standard
Diffusion Policy concatenates the observation into a single ``global_cond`` vector.
The research plan rules that out for this project: the count lives in the sequence, so
pooling ``SCOOP SCOOP SCOOP`` into a fixed vector is exactly the step that destroys
multiplicity and reproduces the methods this work is arguing against.

So the log enters as a **sequence of tokens the denoiser cross-attends to**, and the
pooled variant is kept as a control rather than discarded:

    cond="crossattn"   observation and log tokens stay a sequence; the action tokens
                       attend to them. Order and multiplicity survive.
    cond="global"      everything is mean-pooled into one vector -- the standard DP
                       recipe, and the one the plan predicts will lose the count.

Running both is the point. "Pooling destroys the count" is a claim this project has
asserted and never tested.

**Read the two modes only within themselves, never across.** ``global`` pools the
*vision* tokens as well as the log, so it is handicapped before any memory is involved:
measured on ButtonUnmask, ``none|global`` is 0.6056 against ``none|crossattn`` 0.5235,
a 15.7% penalty with no memory present at all. Comparing ``log|crossattn`` directly
against ``log|global`` therefore conflates pooling the log with pooling the
observation, and answers neither question.

The valid contrast is the *relative* benefit of memory inside each mode -- log against
that mode's own none. If memory buys less under pooling than under cross-attention,
that is evidence about pooling the log specifically, with the observation handicap
divided out by the shared baseline.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def cosine_betas(T: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal's cosine schedule. Gentler on short chunks than linear."""
    t = torch.linspace(0, T, T + 1) / T
    a = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    a = a / a[0]
    return torch.clip(1 - a[1:] / a[:-1], 0.0001, 0.999)


class TimestepEmbedding(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, d * 4), nn.SiLU(), nn.Linear(d * 4, d))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return self.mlp(torch.cat([ang.cos(), ang.sin()], dim=-1))


class DiffusionPolicy(nn.Module):
    """Denoise an action chunk conditioned on observation and (optionally) the log."""

    def __init__(
        self,
        action_dim: int = 8,
        k: int = 20,
        d_feat: int = 2048,
        n_vis: int = 4,
        state_dim: int = 8,
        vocab: int = 256,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        max_log: int = 64,
        memory: str = "log",
        cond: str = "crossattn",
        n_steps: int = 100,
    ) -> None:
        super().__init__()
        if cond not in ("crossattn", "global"):
            raise ValueError("cond must be 'crossattn' or 'global'")
        self.k, self.action_dim = k, action_dim
        self.memory, self.cond = memory, cond
        self.n_steps, self.max_log, self.vocab = n_steps, max_log, vocab

        self.act_in = nn.Linear(action_dim, d_model)
        self.act_pos = nn.Embedding(k, d_model)
        self.t_emb = TimestepEmbedding(d_model)

        self.vis_in = nn.Sequential(nn.LayerNorm(d_feat), nn.Linear(d_feat, d_model))
        self.state_in = nn.Linear(state_dim, d_model)

        self.uses_tokens = memory in ("log", "wrong", "shuffled")
        if self.uses_tokens:
            self.tok_emb = nn.Embedding(vocab + 1, d_model)   # +1 padding
            self.log_pos = nn.Embedding(max_log, d_model)
        elif memory == "count":
            self.count_in = nn.Sequential(
                nn.Linear(vocab, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
            )

        layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_model * 4, batch_first=True, norm_first=True
        )
        self.dec = nn.TransformerDecoder(layer, n_layers)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, action_dim))

        betas = cosine_betas(n_steps)
        alphas = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cum", alphas)

    # ----------------------------------------------------------------- context
    def context(self, feat, state, log_tokens=None, log_len=None, counts=None):
        """The tokens the action chunk is conditioned on, plus a padding mask."""
        toks = [self.vis_in(feat), self.state_in(state).unsqueeze(1)]
        mask = [torch.zeros(feat.shape[0], feat.shape[1] + 1, dtype=torch.bool,
                            device=feat.device)]
        if self.uses_tokens:
            b, L = log_tokens.shape
            pos = torch.arange(L, device=log_tokens.device).unsqueeze(0).expand(b, L)
            toks.append(self.tok_emb(log_tokens) + self.log_pos(pos))
            pad = torch.arange(L, device=log_tokens.device).unsqueeze(0) >= log_len.unsqueeze(1)
            # An all-padded log would make a row's attention undefined; keep slot 0
            # live so an empty log is "nothing has happened yet", not a NaN.
            pad = pad.clone()
            pad[:, 0] = False
            mask.append(pad)
        elif self.memory == "count":
            toks.append(self.count_in(counts).unsqueeze(1))
            mask.append(torch.zeros(feat.shape[0], 1, dtype=torch.bool, device=feat.device))
        ctx = torch.cat(toks, dim=1)
        pad_mask = torch.cat(mask, dim=1)

        if self.cond == "global":
            # The control: collapse everything to one vector. This is what standard
            # Diffusion Policy does, and what the plan predicts will lose the count.
            keep = (~pad_mask).float().unsqueeze(-1)
            pooled = (ctx * keep).sum(1, keepdim=True) / keep.sum(1, keepdim=True).clamp(min=1)
            return pooled, torch.zeros(ctx.shape[0], 1, dtype=torch.bool, device=ctx.device)
        return ctx, pad_mask

    # ----------------------------------------------------------------- forward
    def forward(self, noisy, t, feat, state, log_tokens=None, log_len=None, counts=None):
        b = noisy.shape[0]
        pos = torch.arange(self.k, device=noisy.device).unsqueeze(0).expand(b, self.k)
        x = self.act_in(noisy) + self.act_pos(pos) + self.t_emb(t).unsqueeze(1)
        ctx, pad = self.context(feat, state, log_tokens, log_len, counts)
        h = self.dec(x, ctx, memory_key_padding_mask=pad)
        return self.out(h)

    # ----------------------------------------------------------------- train/sample
    def loss(self, actions, feat, state, log_tokens=None, log_len=None, counts=None):
        b = actions.shape[0]
        t = torch.randint(0, self.n_steps, (b,), device=actions.device)
        noise = torch.randn_like(actions)
        a = self.alphas_cum[t].view(b, 1, 1)
        noisy = a.sqrt() * actions + (1 - a).sqrt() * noise
        pred = self(noisy, t, feat, state, log_tokens, log_len, counts)
        return nn.functional.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, feat, state, log_tokens=None, log_len=None, counts=None,
               steps: int = 20, clip: float = 1.0):
        """DDIM, deterministic (eta=0), so the comparison is not sampling noise.

        ``x0`` is clipped every step, and that is not cosmetic. At the first step
        ``alphas_cum`` is 2.4e-7, so ``x0 = (x - sqrt(1-a) eps) / sqrt(a)`` multiplies
        any error in the predicted noise by about 2000x. Without the clip this produced
        sampled L1 of 83-137 against a predict-the-mean reference of 0.60 -- garbage,
        not a weak policy. Diffusion Policy normalises actions to [-1, 1] precisely so
        that this clip is principled, and the caller is expected to do the same.
        """
        b = feat.shape[0]
        x = torch.randn(b, self.k, self.action_dim, device=feat.device)
        ts = torch.linspace(self.n_steps - 1, 0, steps).long().to(feat.device)
        for i, t in enumerate(ts):
            tb = t.expand(b)
            eps = self(x, tb, feat, state, log_tokens, log_len, counts)
            a = self.alphas_cum[t]
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-clip, clip)
            if i + 1 < len(ts):
                a_next = self.alphas_cum[ts[i + 1]]
                # Recompute eps from the clipped x0 so the step stays self-consistent.
                eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
                x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
            else:
                x = x0
        return x
