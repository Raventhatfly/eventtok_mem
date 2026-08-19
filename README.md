# SpikeBank — STDP-trained multi-timescale spiking memory for Diffusion Policies

A visuomotor policy sees the world as a *stream*. Backprop-through-time over that stream is
expensive, non-causal, and impossible to run at deployment. This project gives a robot policy a
**spiking memory bank** written **online and gradient-free** by STDP, and read by a small learned
head that conditions an otherwise-standard **Diffusion Policy**.

The bank is a set of LIF populations with **heterogeneous membrane decay constants** spanning
three decades, so one module holds three physically distinct kinds of memory:

| substrate | variable | timescale | memory type | plasticity |
|---|---|---|---|---|
| membrane potential | `V` | 0.1–1 s | **iconic** — what I am seeing now | none (dynamics) |
| adaptation / STP | `a` | 1–60 s | **working** — what happened this episode | activity-driven, decays |
| synaptic weights | `W` | episode → dataset | **semantic** — what usually follows what | **STDP** |

Because no gradient crosses the spike layer, the memory trace for a whole dataset can be
**precomputed and cached** — training a memory-augmented Diffusion Policy costs about what
training a memoryless one costs, at any history length.

## Docs

| file | contents |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | the architecture, the maths, the training protocol, the experiment plan, the risks |
| [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md) | what already exists (a lot), where the real gap is, and what **not** to call this |
| [`docs/MECHANISMS.md`](docs/MECHANISMS.md) | the ten literature findings that constrain the build, plus hard capacity ceilings |
| [`docs/STDP_RULES.md`](docs/STDP_RULES.md) | what makes STDP work, its honest ceilings, and the post-mortem on E2 |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | E1: multi-timescale retention, measured. E2: the STDP control, still open |
| [`docs/DP_INTEGRATION.md`](docs/DP_INTEGRATION.md) | verified Diffusion Policy injection points, and the S4D-Real warning |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | what to do next, in order, with the open decisions |

## Code

```
spikebank/
  neurons.py         heterogeneous ALIF bank, one-clock (1 SNN step == 1 control step)
  encoder.py         novelty-gated sparse spike encoder (predictive residual + k-WTA lift)
  stdp.py            trace STDP with per-bank windows, soft bounds, L1 synaptic scaling
  test_retention.py     E1 -- does the tau-hierarchy actually retain?
  test_stdp_control.py  E2 -- does STDP beat frozen-random recurrence?
```

```bash
python -m spikebank.test_retention
python -m spikebank.test_stdp_control
```

## Status

Design + reference implementation + two measured results. **Not yet integrated with a Diffusion
Policy.**

- **E1 established the core claim in miniature:** the tau-spectrum genuinely composes — the
  whole-bank readout recovers a cue at lag 20 (0.98) where no single sub-bank exceeds 0.41. But
  the horizon is ~2 s, not the ~60 s the slowest bank nominally implies: **interference, not leak,
  is the binding constraint.** Decay alone does not buy long memory.
- **E2, the go/no-go control, is still open.** Neither STDP nor frozen-random recurrence clears
  chance on the synthetic order-memory task, so the comparison measures nothing yet. Two rounds of
  fixes (fan-in sparsity, then the Nessler-compliant `exp(-w)` rule) removed real pathologies
  without changing that. `ROADMAP.md` Phase 0 is the blocking work.

Nothing here should be read as evidence for or against STDP yet.

**Open design question, flagged loudly:** a bank differing only in membrane decay is `S4D-Real`,
the weakest state-space initialisation. Timescale diversity in the SSM literature lives in the
per-channel timestep `Δ` and in oscillation frequency, not in the decay rate. `DP_INTEGRATION.md`
§0 has the fix — it is small, but it should be settled before the Phase 1 control.

## Where this sits

Target benchmark is **RoboMemArena** (26 memory-centric LIBERO-compatible manipulation tasks,
CSR/TSR). The memory-probing evaluation reuses **MemProbe**'s deterministic trajectory-derived
facts as linear-probe targets, giving a per-sub-bank memory-retention curve on real robot data.
