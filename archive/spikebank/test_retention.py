"""Unit test for the central claim: different tau sub-banks retain a cue for
different lags, i.e. the bank really is a multi-timescale memory.

Protocol: present one of C cue patterns at t=0, then feed unrelated random
observations for T steps. At each lag, fit a linear probe from each sub-bank's
readout to the cue identity. A working bank shows retention curves that decay
at rates ordered by tau.

Run:  python -m spikebank.test_retention
"""
from __future__ import annotations

import torch

from .encoder import NoveltySparseEncoder
from .neurons import DEFAULT_BANKS, SpikeBank


def linear_probe(h_tr, y_tr, h_te, y_te, n_cls, ridge=1e-2):
    """Closed-form least-squares probe to one-hot targets -> test accuracy."""
    X = torch.cat([h_tr, torch.ones(len(h_tr), 1)], 1)
    Y = torch.nn.functional.one_hot(y_tr, n_cls).float()
    W = torch.linalg.solve(X.T @ X + ridge * torch.eye(X.shape[1]), X.T @ Y)
    Xte = torch.cat([h_te, torch.ones(len(h_te), 1)], 1)
    return ((Xte @ W).argmax(1) == y_te).float().mean().item()


def main(B=512, T=200, d=64, n_cls=8, seed=0):
    torch.manual_seed(seed)
    enc = NoveltySparseEncoder(d_in=d, n_out=1024, k=32, seed=seed)
    bank = SpikeBank(n_in=1024, banks=DEFAULT_BANKS, dt=0.1, k_active=16, k_in=32, seed=seed)

    cues = torch.randn(n_cls, d) * 3.0
    y = torch.randint(0, n_cls, (B,))
    st = bank.init_state(B)
    zh = enc.init_state(B, d)

    lags = [1, 2, 5, 10, 20, 50, 100, 200]
    snap = {}
    for t in range(T + 1):
        z = cues[y] if t == 0 else torch.randn(B, d)      # cue at t=0, then noise
        x, zh, _ = enc(z, zh)
        _, st = bank.step(x, st)
        if t in lags:
            snap[t] = bank.readout(st).clone()

    sizes, tr = bank.sizes, slice(0, B // 2)
    te = slice(B // 2, B)
    names = [f"tau={b.tau_mem:>5.1f}s" for b in DEFAULT_BANKS]
    print(f"\ncue-identity linear-probe accuracy (chance = {1/n_cls:.3f})")
    print("lag(steps) " + " ".join(f"{n:>12s}" for n in names) + "   all-banks")
    for L in lags:
        h = snap[L]
        lp, ad = h.split(bank.n_total, dim=1)
        row = []
        for i in range(len(sizes)):
            o = sum(sizes[:i])
            hb = torch.cat([lp[:, o:o + sizes[i]], ad[:, o:o + sizes[i]]], 1)
            row.append(linear_probe(hb[tr], y[tr], hb[te], y[te], n_cls))
        allb = linear_probe(h[tr], y[tr], h[te], y[te], n_cls)
        print(f"{L:>10d} " + " ".join(f"{a:>12.3f}" for a in row) + f"   {allb:>8.3f}")


if __name__ == "__main__":
    main()
