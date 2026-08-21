"""Heterogeneous multi-timescale ALIF bank (one-clock: 1 SNN step == 1 control step).

Design ref: docs/DESIGN.md sections 4.3, 5.1, 5.2.

The bank holds THREE memory substrates at three timescales:
  V  membrane potential      tau_mem  in [0.1s, 60s]   -> iconic / working
  a  adaptation variable     tau_adp  = 5 * tau_mem    -> working
  W  recurrent weights       (STDP, see stdp.py)       -> semantic

No autograd flows through the spike nonlinearity. The bank is run under
``torch.no_grad()``; the downstream readout consumes ``h_t.detach()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class BankSpec:
    """One sub-bank: N neurons sharing a nominal timescale."""

    n_neurons: int
    tau_mem: float           # seconds
    tau_adp_mult: float = 5.0
    het_octaves: float = 0.5  # per-neuron log2-uniform jitter around tau_mem


DEFAULT_BANKS = (
    BankSpec(256, 0.1),
    BankSpec(256, 0.5),
    BankSpec(256, 2.0),
    BankSpec(256, 10.0),
    BankSpec(256, 60.0),
)


class SpikeBank(torch.nn.Module):
    """M sub-banks of adaptive LIF neurons with heterogeneous decay constants.

    Args:
        n_in: width of the sparse input code x_t (from encoder.py).
        banks: nominal timescales, one entry per sub-bank.
        dt: control period in seconds (0.1 == 10 Hz).
        k_active: k-WTA winners per sub-bank per step (lateral inhibition).
        k_in: expected number of active input lines per step (encoder's k). Scales
            w_in so one input volley gives a unit membrane deflection regardless of
            the sub-bank's tau -- without this, fast banks never fire and slow banks
            fire only after integrating for hundreds of steps.
        target_rate: homeostatic duty-cycle target, used by the threshold controller.
    """

    def __init__(
        self,
        n_in: int,
        banks: tuple[BankSpec, ...] = DEFAULT_BANKS,
        dt: float = 0.1,
        k_active: int = 16,
        k_in: int = 32,
        target_rate: float = 0.03,
        theta0: float = 1.0,
        gamma: float = 1.8,
        w_in_density: float = 0.05,
        seed: int = 0,
    ) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.banks, self.dt, self.k_active = banks, dt, k_active
        self.target_rate, self.theta0, self.gamma = target_rate, theta0, gamma
        self.n_total = sum(b.n_neurons for b in banks)
        self.sizes = [b.n_neurons for b in banks]

        # --- per-neuron heterogeneous time constants (log-uniform around nominal) ---
        tau_mem, tau_adp = [], []
        for b in banks:
            jitter = (torch.rand(b.n_neurons, generator=g) * 2 - 1) * b.het_octaves
            t_m = b.tau_mem * torch.pow(2.0, jitter)
            tau_mem.append(t_m)
            tau_adp.append(t_m * b.tau_adp_mult)
        tau_mem = torch.cat(tau_mem)
        tau_adp = torch.cat(tau_adp)
        self.register_buffer("beta", torch.exp(-dt / tau_mem))    # membrane decay
        self.register_buffer("rho", torch.exp(-dt / tau_adp))     # adaptation decay
        self.register_buffer("tau_mem", tau_mem)

        # --- sparse fixed input projection; recurrent weights are STDP-plastic ---
        w_in = (torch.rand(n_in, self.n_total, generator=g) < w_in_density).float()
        w_in *= torch.rand(n_in, self.n_total, generator=g)
        # unit-deflection scaling: E[drive] ~= 1.0 for a k_in-hot input code
        w_in = w_in / w_in.sum(0, keepdim=True).clamp_min(1e-6) * (n_in / max(k_in, 1))
        self.register_buffer("w_in", w_in)
        # block-diagonal recurrence: no cross-bank recurrence, so each timescale
        # stays a clean, independently interpretable memory channel.
        self.w_rec = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(n, n), requires_grad=False) for n in self.sizes]
        )
        self.register_buffer("theta_bias", torch.zeros(self.n_total))  # homeostatic offset
        self.register_buffer("rate_ema", torch.full((self.n_total,), target_rate))

    # ------------------------------------------------------------------ state
    def init_state(self, batch: int, device=None) -> dict[str, torch.Tensor]:
        device = device or self.beta.device
        z = lambda: torch.zeros(batch, self.n_total, device=device)
        return {"V": z(), "a": z(), "s": z(), "lp": z()}

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, x_t: torch.Tensor, state: dict) -> tuple[torch.Tensor, dict]:
        """Advance one control step. x_t: (B, n_in) sparse binary code."""
        V, a, s_prev = state["V"], state["a"], state["s"]

        drive = x_t @ self.w_in
        rec = torch.cat(
            [sp @ w for sp, w in zip(s_prev.split(self.sizes, dim=1), self.w_rec)], dim=1
        )
        theta_prev = self.theta0 + self.gamma * a + self.theta_bias

        # reset by subtraction: keeps sub-threshold information, unlike reset-to-zero
        V = self.beta * V + drive + rec - theta_prev * s_prev
        a = self.rho * a + s_prev
        theta = self.theta0 + self.gamma * a + self.theta_bias

        s = (V > theta).float()
        s = self._kwta(V - theta, s)

        # homeostasis: slow per-neuron threshold offset toward target duty cycle
        self.rate_ema.mul_(0.99).add_(0.01 * s.mean(0))
        self.theta_bias.add_(0.5 * (self.rate_ema - self.target_rate))

        lp = self.beta * state["lp"] + (1 - self.beta) * s   # tau-matched readout filter
        return s, {"V": V, "a": a, "s": s, "lp": lp}

    def _kwta(self, margin: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Keep only the top-k supra-threshold neurons within each sub-bank."""
        out = []
        for mg, sp in zip(margin.split(self.sizes, dim=1), s.split(self.sizes, dim=1)):
            k = min(self.k_active, mg.shape[1])
            idx = mg.topk(k, dim=1).indices
            mask = torch.zeros_like(sp).scatter_(1, idx, 1.0)
            out.append(sp * mask)
        return torch.cat(out, dim=1)

    # ------------------------------------------------------------------ readout
    @staticmethod
    def readout(state: dict) -> torch.Tensor:
        """h_t consumed by the (backprop-trained) readout head. Detached by construction."""
        return torch.cat([state["lp"], state["a"]], dim=1)

    @property
    def readout_dim(self) -> int:
        return 2 * self.n_total
