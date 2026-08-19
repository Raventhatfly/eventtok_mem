"""Trace-based pair STDP on the bank's recurrent weights.

Design ref: docs/DESIGN.md section 5.3. What this learns is TRANSITIONS, not
features: if code A reliably precedes code B, W[A, B] grows, so presenting A
pre-activates B. That is what makes the readout memory-like rather than a
low-pass filter.

Three stabilisers are mandatory and all three are here:
  k-WTA (in neurons.py) / homeostasis (in neurons.py) / L1 row normalisation.

The STDP window scales with each sub-bank's tau -- a single global window
across banks whose dynamics differ by 1000x is a bug, not a simplification.
"""
from __future__ import annotations

import torch


class STDP:
    def __init__(
        self,
        bank,
        eta: float = 2e-3,
        a_minus: float = 0.6,
        w_max: float = 1.0,
        trace_steps: float = 3.0,
        fan_in: int = 24,
    ) -> None:
        self.bank, self.eta, self.a_minus, self.w_max = bank, eta, a_minus, w_max
        # hard fan-in cap: without it, potentiation + L1 normalisation drives W to ~100%
        # density and the "learned" recurrence degenerates into a uniform low-pass filter
        # (see docs/EXPERIMENTS.md E2).
        self.fan_in = fan_in
        # per-sub-bank trace decay, tied to that bank's nominal membrane tau
        self.alpha = [
            float(torch.exp(torch.tensor(-1.0 / (trace_steps * b.tau_mem / bank.dt))))
            for b in bank.banks
        ]
        self.reset()

    def reset(self) -> None:
        self.p = [None] * len(self.bank.sizes)   # presynaptic eligibility traces
        self.q = [None] * len(self.bank.sizes)   # postsynaptic traces

    @torch.no_grad()
    def update(self, s_t: torch.Tensor, modulator: float = 1.0) -> None:
        """One online STDP update. s_t: (B, n_total) spikes at this control step.

        ``modulator`` is the optional third factor (phase 3): pass a scalar
        reward / negative-loss signal to gate plasticity. Default 1.0 = pure
        two-factor Hebbian STDP, no task feedback, no gradients.
        """
        for m, (sp, w) in enumerate(zip(s_t.split(self.bank.sizes, dim=1), self.bank.w_rec)):
            a = self.alpha[m]
            if self.p[m] is None:
                self.p[m] = torch.zeros_like(sp)
                self.q[m] = torch.zeros_like(sp)
            p, q = self.p[m], self.q[m]

            # potentiation: pre-trace (past) x post-spike (now)  -> causal A->B
            pot = p.t() @ sp
            # depression: pre-spike (now) x post-trace (past)    -> anticausal B->A
            dep = sp.t() @ q

            dw = self.eta * modulator * ((self.w_max - w) * pot - self.a_minus * w * dep)
            w.add_(dw / sp.shape[0])
            w.clamp_(0.0, self.w_max)
            w.fill_diagonal_(0.0)                            # no self-excitation
            if self.fan_in and self.fan_in < w.shape[0]:     # keep recurrence sparse
                keep = w.topk(self.fan_in, dim=0).indices
                mask = torch.zeros_like(w).scatter_(0, keep, 1.0)
                w.mul_(mask)
            w.div_(w.sum(0, keepdim=True).clamp_min(1e-3))   # L1 synaptic scaling

            self.p[m] = a * p + sp
            self.q[m] = a * q + sp
