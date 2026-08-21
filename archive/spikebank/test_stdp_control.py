"""E2 -- the go/no-go control (DESIGN.md 8.1): does STDP beat frozen-random recurrence?

Most of the memory in a spiking reservoir comes from the leaky dynamics, not from the
learning rule. If STDP does not beat a random recurrent matrix at equal size, the STDP
story is decoration.

Stream: each episode has a latent TASK id. A task fixes a phase-transition order over K
shared phases; each phase emits a noisy pattern. The per-frame observations are drawn
from the SAME pool for every task -- only the ORDER differs. So the task is decodable
only from remembered transitions, never from the current frame. That is exactly what
STDP on recurrent weights is supposed to learn.

Probe target = task id at increasing lag. Chance = 1/n_tasks.
"""
from __future__ import annotations

import torch

from .encoder import NoveltySparseEncoder
from .neurons import DEFAULT_BANKS, SpikeBank
from .stdp import STDP
from .test_retention import linear_probe


def make_stream(n_tasks, n_phase, d, T, B, g):
    """Returns z: (T, B, d) and task ids y: (B,). Frames are task-independent in
    marginal distribution; only the transition ORDER carries the task identity."""
    proto = torch.randn(n_phase, d, generator=g) * 3.0
    orders = torch.stack([torch.randperm(n_phase, generator=g) for _ in range(n_tasks)])
    y = torch.randint(0, n_tasks, (B,), generator=g)
    z, pos = [], torch.zeros(B, dtype=torch.long)
    dwell = 4                                     # steps spent in each phase
    for t in range(T):
        ph = orders[y, (pos // dwell) % n_phase]
        z.append(proto[ph] + 0.5 * torch.randn(B, d, generator=g))
        pos += 1
    return torch.stack(z), y


def run(bank, enc, z, train_stdp=None, lags=()):
    T, B, d = z.shape
    st, zh, snap = bank.init_state(B), enc.init_state(B, d), {}
    for t in range(T):
        x, zh, _ = enc(z[t], zh)
        s, st = bank.step(x, st)
        if train_stdp is not None:
            train_stdp.update(s)
        if t in lags:
            snap[t] = bank.readout(st).clone()
    return snap


def main(n_tasks=6, n_phase=6, d=64, T=120, B=512, epochs=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    lags = [10, 20, 40, 80, 119]
    z_tr, y_tr = make_stream(n_tasks, n_phase, d, T, B, g)
    z_te, y_te = make_stream(n_tasks, n_phase, d, T, B, g)

    results = {}
    for arm in ("random", "stdp"):
        torch.manual_seed(seed)
        enc = NoveltySparseEncoder(d_in=d, n_out=1024, k=32, seed=seed)
        bank = SpikeBank(n_in=1024, dt=0.1, k_active=16, k_in=32, seed=seed)
        if arm == "random":                       # frozen random recurrence, same scale
            gg = torch.Generator().manual_seed(seed + 1)
            for w in bank.w_rec:
                r = (torch.rand(w.shape, generator=gg) < 0.1).float() * torch.rand(w.shape, generator=gg)
                r.fill_diagonal_(0.0)
                w.copy_(r / r.sum(0, keepdim=True).clamp_min(1e-3))
            learner = None
        else:                                     # unsupervised STDP passes over the stream
            learner = STDP(bank, eta=5e-3)
            for _ in range(epochs):
                learner.reset()
                run(bank, enc, z_tr, train_stdp=learner)
            learner = None                        # freeze before evaluation

        snap_tr = run(bank, enc, z_tr, lags=lags)
        snap_te = run(bank, enc, z_te, lags=lags)
        results[arm] = {L: linear_probe(snap_tr[L], y_tr, snap_te[L], y_te, n_tasks) for L in lags}
        results[arm]["density"] = float(sum(w.gt(1e-4).float().mean() for w in bank.w_rec) / len(bank.w_rec))

    print(f"\nE2  task-id decodable only from remembered ORDER   chance = {1/n_tasks:.3f}")
    print(f"{'lag':>6} {'frozen-random':>15} {'STDP':>10}   delta")
    for L in lags:
        r, s = results["random"][L], results["stdp"][L]
        print(f"{L:>6} {r:>15.3f} {s:>10.3f}   {s - r:+.3f}")
    print(f"\nrecurrent weight density  random={results['random']['density']:.3f}  "
          f"stdp={results['stdp']['density']:.3f}")


if __name__ == "__main__":
    main()
