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

---

## E2 — STDP vs. frozen-random recurrence, first pass (2026-08-18)

**Question.** The go/no-go control of `DESIGN.md` §8.1. Does STDP beat a frozen-random recurrent
matrix at equal size?

**Protocol** (`spikebank/test_stdp_control.py`). Each episode has a latent task id. A task fixes a
*phase-transition order* over 6 shared phases; each phase emits a noisy pattern. Every task draws
frames from the **same pool** — only the order differs — so the task is decodable *only* from
remembered transitions, never from the current frame. This is precisely what STDP on recurrent
weights is supposed to learn. 3 unsupervised STDP passes, then freeze, then linear-probe the task
id at increasing lag. 6 tasks, chance = 0.167.

**Result.**

| lag | frozen-random | STDP | Δ |
|---|---|---|---|
| 10 | 0.119 | 0.238 | +0.119 |
| 20 | 0.186 | 0.111 | −0.074 |
| 40 | 0.238 | 0.252 | +0.014 |
| 80 | 0.326 | 0.037 | −0.289 |
| 119 | 0.146 | 0.225 | +0.078 |

Recurrent weight density: random 0.099 → **STDP 0.996**.

**Reading it honestly.**

- **This control is not yet decisive, and should not be reported as one.** Both arms hover near
  chance (0.12–0.33 against 0.167), with signs of Δ flipping across lags — that is noise, not a
  result. Neither arm actually solves the task, so the comparison is underpowered.
- **The informative signal is the weight densification.** STDP drove the recurrent matrix from
  10 % to **99.6 % density**. That is the textbook pathology: potentiation plus L1 row
  normalisation spreads weight across all inputs and destroys selectivity, so the "learned"
  recurrence is close to a uniform low-pass filter — which is *worse* than a sparse random matrix
  because it is less non-normal (cf. Ganguli et al. 2008, `MECHANISMS.md` §2).
- Diagnosis matches Zenke, Gerstner & Ganguli (2017): the stabilising machinery is on the wrong
  timescale relative to the Hebbian instability. `stdp.py` has soft bounds, an L1 row constraint
  and a depression term, but **no sparsity constraint on `W` itself** and no fast inhibitory
  plasticity.

**Fixes to try before calling the control, in order.**
1. **Sparsify `W`** — hard top-k per postsynaptic neuron after each update, or a pruning threshold.
   Keeping recurrence at ~5–10 % density is the single most likely fix.
2. **Raise the depression term** `a_minus` (currently 0.6) — the potentiation/depression balance is
   what sets the stationary weight distribution (Fusi & Abbott 2007).
3. **Non-normal chain wiring** (`MECHANISMS.md` §2) — currently block-diagonal, i.e. exactly the
   normal-network regime with an O(1) memory ceiling.
4. **Delta-rule writes** instead of plain additive potentiation (Schlag et al. 2021).
5. Make the task easier first (more phases, longer dwell) so *some* arm clears chance — a control
   between two failing arms measures nothing.

**Status: the go/no-go question is still open.** That is the right thing to know on day one, and
it is exactly why `DESIGN.md` puts this experiment before any Diffusion Policy integration.

### E2b — with fix #1 applied (hard fan-in cap on `W`)

Added a hard fan-in cap (top-24 presynaptic partners per postsynaptic neuron) to `stdp.py`.

| lag | frozen-random | STDP | Δ |
|---|---|---|---|
| 10 | 0.119 | 0.176 | +0.057 |
| 20 | 0.186 | 0.182 | −0.004 |
| 40 | 0.238 | 0.086 | −0.152 |
| 80 | 0.326 | 0.168 | −0.158 |
| 119 | 0.146 | 0.158 | +0.012 |

Density: random 0.099, **STDP 0.094** — the pathology is fixed.

**But the control is still inconclusive, and for the reason predicted: neither arm clears chance.**
The probe cannot decode task identity from either bank, so there is nothing for the comparison to
measure. Fix #5 (make the task solvable by *something* first) is now the blocking item, ahead of
fixes #2–#4. Concretely: longer phase dwell, more phases, a larger bank, and a sanity ceiling
(probe an explicit frame-history buffer) to confirm the task is decodable at all before comparing
memory architectures on it.

**Do not read either E2 table as evidence against STDP.** They are evidence that the harness is
not yet sensitive enough to answer the question.

### E2c — with the Nessler-compliant rule

Replaced the linear `(w_max − w)` bound with **`exp(−w)` potentiation** and set LTD/LTP to
**1:100** (Diehl & Cook's released values). See `STDP_RULES.md` §0 for why both matter.

| lag | frozen-random | STDP | Δ |
|---|---|---|---|
| 10 | 0.119 | 0.131 | +0.012 |
| 20 | 0.186 | 0.154 | −0.031 |
| 40 | 0.238 | 0.221 | −0.018 |
| 80 | 0.326 | 0.219 | −0.107 |
| 119 | 0.146 | 0.176 | +0.029 |

Unchanged conclusion, as predicted: **the harness, not the rule, is the blocker.** Neither arm
clears chance, so no learning-rule change can show up. The remaining preconditions from
`STDP_RULES.md` §0 are still unmet — **soft/stochastic WTA with a temperature** (we use hard
top-k, and Nessler's theorem needs the soft version because the WTA *is* the normalisation) and
**population-coded inputs**. Those are changes to `neurons.py` and `encoder.py`, not `stdp.py`.

**E3 (next session), in order:**
1. Add a sanity ceiling — probe an explicit frame-history buffer — to prove the task is decodable
   at all. **Do not compare memory architectures on a task nothing solves.**
2. Softmax WTA with temperature `T` (SoftHebb) replacing hard top-k.
3. Population-code the proprio/low-dimensional channel.
4. Non-normal chain wiring between sub-banks (`MECHANISMS.md` §2).
5. Only then re-run the STDP-vs-random control.
