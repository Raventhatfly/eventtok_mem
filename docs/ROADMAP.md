# Roadmap — what to do next, in order

*Written 2026-08-18 after the literature sweep and E1/E2. Ordered so the cheapest experiment that
could kill the project runs first.*

## Phase 0 — make the harness sensitive (days, blocking everything)

E2 could not answer its own question because neither arm cleared chance. Before any architectural
comparison:

1. **Sanity ceiling.** Probe an explicit frame-history buffer on the same synthetic task. If *that*
   cannot decode the task id, the task is broken, not the model.
2. **Soft/stochastic WTA with temperature `T`** replacing hard top-k in `neurons.py`. Nessler's
   EM theorem needs the soft version — the WTA *is* the normalisation, not a sparsity gadget.
   SoftHebb also reports soft > hard on both accuracy and noise robustness.
3. **Population-code the low-dimensional channel** (proprioception). The theory assumes it.
4. **Non-normal chain wiring** fast→medium→slow between sub-banks, replacing block-diagonal.

**Exit criterion:** some configuration decodes the synthetic task well above chance.

## Phase 1 — the go/no-go control (week 1)

**STDP vs. frozen-random recurrence, at equal size.** Re-run E2 on the now-sensitive harness.

- **If STDP wins** → proceed to Phase 2 with the STDP story intact.
- **If STDP ties** → re-scope to "heterogeneous-timescale spiking memory for policy conditioning".
  Still honest, still publishable, and cheaper. `PRIOR_ART.md` §5 shows the τ-heterogeneity
  mechanism is established but has never been applied to policy conditioning.
- **If STDP loses** → the recurrence is hurting; fall back to a feedforward multi-τ bank.

Add the third baseline the literature demands: **the same bank trained by surrogate-gradient BPTT**.
Without it, "we used STDP" reads as "we used a worse learning rule". Expect ~2.4 pts below BPTT
*with* recurrence and ~13–17 pts *without* it — those SHD numbers are the calibration.

## Phase 2 — real data before real policy (weeks 2–3)

Run the bank over RoboMemArena episodes and produce the **memory-retention curve per sub-bank**,
with **MemProbe** facts as linear-probe targets. This is the most convincing figure available and
it needs no Diffusion Policy at all.

Measure three things the literature says will break:
- **Feature coherence** between bank weight vectors. Falez et al. saw 0.999 — if ours approaches
  1.0, the inhibition is decorative and sparsity is hiding it.
- **Novelty-gate selectivity** on real frames. E1 showed we cannot assume redundant frames get
  gated out; on robot data most frames *are* near-redundant, so measure the actual write rate.
- **Dead/dominating neurons** under temporally correlated input. Robot streams are far worse than
  MNIST for this — one neuron can win thousands of consecutive updates. Shorten `tc_theta` to
  10²–10³ s and consider a short shuffle buffer.

## Phase 3 — policy integration (days, once Phase 2 is convincing)

One-line `global_cond` concat into Diffusion Policy (`DESIGN.md` §6.2), plus the cache trick
(§7) so training cost stays flat in history length. Then the full RoboMemArena sweep.

Baselines are **PTP, VQ-Memory, DSSP, MemoryVAM** — not vanilla DP. **DSSP is the head-to-head**:
an SSM history encoder vs. our spiking bank is the cleanest comparison, since both are
diagonal-decay memories differing mainly in how the recurrence is learned.

## Phase 4 — the claim that is actually ours

Deploy-time plasticity: keep STDP on during evaluation and show within-episode recall improving
over the episode, with no backward pass. This is claim 1 in `DESIGN.md` §2 and the one an SSM
architecturally cannot make.

## Open decisions

| decision | options | current lean |
|---|---|---|
| τ initialisation | hand-picked log-grid vs. **LMU/HiPPO basis** | LMU basis, log-grid as ablation |
| long-timescale substrate | one long τ vs. **DEXAT dual decay** vs. adaptive threshold | dual decay + SFA; a single τ=60 s membrane is badly conditioned |
| competition | hard k-WTA vs. **softmax with temperature** | softmax — required by the theory, better empirically |
| third factor | phase 3 vs. **from day one** | day one. Mozafari: 66 % → 88.4 % on NORB. STDP learns what *repeats*, and on a robot that is floor texture and idle proprioception |
| one population or two | single bank vs. **sparse-coding readout + EM-style associative store** | two — one layer cannot be both a sparse code and a prototype memory |
| encoding | Poisson rate vs. **rank-order/TTFS** | rank-order: scale-invariant, so encoder drift across scenes is free robustness |

## What not to do

- Don't call it "Spiking Diffusion Policy" — that name and architecture are taken.
- Don't lead with energy. Measured neuromorphic wins are ~17×, and SpikeVLA's honest number is
  −66 %; the 94–99 % figures in the spiking-DP papers are 45 nm SOP estimates.
- Don't spike the perception front end. SoftHebb's ImageNet top-1 of 27.3 % is the ceiling for
  feedback-free local learning. Keep the encoder frozen and pretrained.
- Don't build the deep STDP stack. Usable depth is ~3–5 layers and activity dies across them.
  Prefer **one wide, strongly recurrent layer** — recurrence is where local rules are competitive.
