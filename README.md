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
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | E1: multi-timescale retention, measured |

## Code

```
spikebank/
  neurons.py         heterogeneous ALIF bank, one-clock (1 SNN step == 1 control step)
  encoder.py         novelty-gated sparse spike encoder (predictive residual + k-WTA lift)
  stdp.py            trace STDP with per-bank windows, soft bounds, L1 synaptic scaling
  test_retention.py  the precondition test: does the tau-hierarchy actually retain?
```

```bash
python -m spikebank.test_retention
```

## Status

Design + reference implementation + one measured result (E1). **Not yet integrated with a
Diffusion Policy.** The next milestone is the go/no-go control in `DESIGN.md` §8.1: STDP vs.
frozen-random recurrence. If STDP does not beat a random reservoir, the project re-scopes.

## Where this sits

Target benchmark is **RoboMemArena** (26 memory-centric LIBERO-compatible manipulation tasks,
CSR/TSR). The memory-probing evaluation reuses **MemProbe**'s deterministic trajectory-derived
facts as linear-probe targets, giving a per-sub-bank memory-retention curve on real robot data.
