# Experiment log

## E1 — multi-timescale retention, synthetic cue-recall (2026-08-18)

**Question.** Does a bank of heterogeneous-τ ALIF neurons actually retain a cue for
τ-ordered durations? This is the precondition for everything else in the design.

**Protocol** (`spikebank/test_retention.py`). Present one of 8 random cue patterns at t=0, then
feed 200 steps of unrelated random observations. At each lag, fit a closed-form linear probe from
each sub-bank's readout `[LP(s); a]` to the cue identity. B=512 trials, train/test split by trial.
Chance = 0.125. dt = 0.1 s, N=256/bank, k-WTA = 16, encoder k = 32 of 1024.

**Result.**

| lag (steps) | τ=0.1 s | τ=0.5 s | τ=2 s | τ=10 s | τ=60 s | all banks |
|---|---|---|---|---|---|---|
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 1.000 | 1.000 | 1.000 | 0.996 | 1.000 | 1.000 |
| 5 | 0.934 | 1.000 | 0.918 | 0.816 | 0.746 | 1.000 |
| 10 | 0.227 | 0.812 | 0.668 | 0.598 | 0.590 | **1.000** |
| 20 | 0.109 | 0.309 | 0.406 | 0.344 | 0.375 | **0.984** |
| 50 | 0.156 | 0.109 | 0.148 | 0.207 | 0.293 | 0.148 |
| 100 | 0.141 | 0.125 | 0.164 | 0.184 | 0.121 | 0.109 |
| 200 | 0.129 | 0.152 | 0.156 | 0.137 | 0.156 | 0.086 |

**Reading it honestly.**

1. **The fast bank behaves exactly as predicted** — τ=0.1 s collapses to chance by lag 10,
   while every slower bank is still well above chance. The τ-ordering is real at short lag.
2. **The whole-bank readout is strictly better than any single bank** (1.000 at lag 10, 0.984 at
   lag 20 vs. ≤0.81 and ≤0.41 for the best individual bank). The τ-spectrum genuinely composes:
   this is the multi-timescale claim, and it holds.
3. **But the horizon is ~20 steps (2 s), not the ~600 steps the τ=60 s bank nominally implies.**
   Beyond lag 20 everything is at chance. **The binding constraint is interference, not leak.**
   A slow leaky integrator integrates the *distractors* as faithfully as the cue; a long τ buys
   persistence and equal amounts of noise accumulation. Decay alone does not give long memory.
4. Ordering among the slow banks is non-monotone (τ=0.5 s beats τ=2/10/60 s at lag 10), which is
   the same effect: slower banks have lower per-event gain, so a single cue deflects them less.

**Consequences for the design.** This is the empirical case for every mechanism in
`MECHANISMS.md` beyond plain decay:

- **Non-normal chain wiring** (Ganguli et al. 2008) — the current block-diagonal bank is close to
  the "normal network, O(1) total memory" regime the theory warns about.
- **STDP recurrence** — not yet enabled in this run. If the τ-hierarchy alone tops out at 2 s,
  the long-horizon memory *must* come from learned recurrence / attractor structure, which
  raises the stakes on the §8.1 STDP-vs-random control.
- **Novelty gating must be sharper.** Here the distractors were as novel as the cue, so the gate
  passed everything. On real robot data most frames are near-redundant and the gate should be far
  more selective — but this run shows we cannot rely on that; it needs measuring on RoboMemArena
  data directly.
- **Adaptive threshold / SFA and STP** carry seconds-scale state without paying the
  noise-integration cost, which is precisely the gap this table exposes.

**Also validated:** drive normalisation matters enormously. The first run (input weights
column-normalised to sum 1, no `k_in` scaling) left the bank effectively silent — every probe at
chance for the first 10 steps, and *only* the slow banks eventually fired, after integrating for
~100 steps. Scaling `w_in` by `n_in/k_in` for unit membrane deflection, and raising the
homeostatic rate from 0.01 to 0.5, fixed it. This is a concrete instance of Zenke et al. (2017):
**slow homeostasis cannot stabilise a spiking bank on task-relevant timescales.**

**Next.** E2 = same protocol with STDP enabled vs. frozen-random recurrence (the go/no-go control,
DESIGN.md §8.1). E3 = the same retention curve on real RoboMemArena episodes with MemProbe facts
as the probe target.
