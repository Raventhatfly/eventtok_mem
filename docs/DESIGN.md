# SpikeBank: An STDP-Trained Multi-Timescale Spiking Memory for Diffusion Policies

**Status:** design draft v0.2 — 2026-08-18
**Companion docs:** `PRIOR_ART.md` (positioning), `MECHANISMS.md` (what the literature forces),
`EXPERIMENTS.md` (E1 retention result). **Read the "revisions" box below before implementing.**

> ### Revisions after the literature sweep (v0.1 → v0.2)
> 1. **Block-diagonal recurrence is wrong.** Ganguli et al. (PNAS 2008): normal networks have
>    O(1) *total* memory; only non-normal (feedforward-chain) connectivity is extensive in N.
>    Wire the sub-banks fast→medium→slow, recurrence within-bank only. (§5.1)
> 2. **Do not use a single τ=60 s membrane.** DEXAT (Nat. Commun. 2021): two coupled decays
>    (30 ms + 300 ms) hold 1.2 s better-conditioned than one long τ, which in low precision
>    silently rounds `exp(-dt/τ)` to 1.0. (§5.1)
> 3. **Initialise β from the LMU/HiPPO basis** for a target window θ rather than a hand-picked
>    log-grid; keep the log-grid as an ablation. A spiking LMU on Loihi already exists. (§5.1)
> 4. **Use the delta rule for the associative write**, not plain additive Hebbian — outer-product
>    stores collide past key-dimension items (Schlag et al., ICML 2021). (§5.3)
> 5. **Do not name this "Spiking Diffusion Policy"** — SDP (arXiv:2409.11195) exists and spikes
>    the *denoiser*. Ours is the inverse. See `PRIOR_ART.md` §1.
> 6. **The baselines are PTP / VQ-Memory / DSSP**, not vanilla DP. Beating `DP(To=2)` proves
>    nothing. (§8.2)
> 7. **Empirically (E1): decay alone tops out at ~2 s of retention** — interference, not leak, is
>    the binding constraint. Long-horizon memory must come from learned recurrence, which raises
>    the stakes on the §8.1 control.
**Repo:** `ssn_robotic_memory`
**Target benchmark:** RoboMemArena (26 memory-centric LIBERO-compatible manipulation tasks, CSR/TSR)

---

## 1. One-paragraph thesis

A visuomotor policy sees the world as a *stream*. Backprop-through-time over that stream is
expensive, non-causal, and impossible to run at deployment. We instead give the policy a
**spiking memory bank** that is written **online and gradient-free** by STDP, and **read** by a
small learned linear head that conditions an otherwise-standard **Diffusion Policy**. The bank is
a set of LIF populations with **heterogeneous membrane decay constants** spanning three decades,
so a single module holds iconic (~0.1 s), working (~1–30 s), and semantic (episode/dataset)
memory in three physically distinct state variables. Because no gradient crosses the spike
layer, the memory trace for an entire dataset can be **precomputed and cached**, so training a
memory-augmented Diffusion Policy costs roughly what training a memoryless one costs.

---

## 2. Why this is not just "add an RNN"

The obvious baseline is a GRU / Mamba / transformer over observation history. The SNN bank is
worth building only if it buys something those cannot. Three defensible claims, ranked by how
much I believe them:

| # | Claim | Confidence | How it is tested |
|---|-------|-----------|------------------|
| 1 | **Gradient-free online write.** The bank keeps writing at deployment — new objects, new room layouts, new task instances get encoded without any backward pass. An SSM's memory is frozen the moment training ends. | High — this is architectural, not empirical | Deploy on a held-out RoboMemArena task family with no fine-tuning; measure whether within-episode recall improves over the episode |
| 2 | **Training cost.** No BPTT ⇒ memory features are a *pure function of the past* ⇒ cache `r_t` for the whole dataset once. Memory-augmented training becomes as cheap as memoryless training, at any history length. | High | Wall-clock + VRAM vs. DP+Mamba at equal history length |
| 3 | **Task success on memory tasks.** | **Low, and it must stay low until measured** | RoboMemArena CSR/TSR vs. all baselines |

Claim 3 is the one a reviewer will ask about and the one I am least sure of. The plan in §7 is
built so we learn the answer in week 1, not month 3.

---

## 3. System overview

```
        o_t = (agentview RGB, wrist RGB, ee/gripper/joint state)
                            │
              ┌─────────────▼──────────────┐
              │  frozen perception encoder │   (shared with the DP's own encoder)
              │  DINOv2 / ResNet18 → z_t   │
              └─────────────┬──────────────┘
                            │  z_t ∈ R^d          d ≈ 512
              ┌─────────────▼──────────────┐
   STAGE 1    │  novelty-gated sparse      │   predictive residual + random lift + k-WTA
              │  spike encoder             │   → x_t ∈ {0,1}^{N_in},  ‖x_t‖₀ = k
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────────────────────────────┐
   STAGE 2    │  SPIKE BANK — M sub-banks, heterogeneous τ         │
              │  ┌──────────┬──────────┬──────────┬──────────┐     │
              │  │ τ=0.1 s  │ τ=1 s    │ τ=10 s   │ τ=60 s   │ ... │
              │  │ N=256    │ N=256    │ N=256    │ N=256    │     │
              │  └──────────┴──────────┴──────────┴──────────┘     │
              │  recurrent W^rec learned ONLINE by STDP            │
              │  + k-WTA lateral inhibition + homeostasis          │
              └─────────────┬──────────────────────────────────────┘
                            │  spikes s_t, adaptation a_t, filtered traces
              ┌─────────────▼──────────────┐
   STAGE 3    │  learned linear readout gφ │   ← backprop STOPS here (no BPTT, no surrogate grad)
              │  → r_t ∈ R^{d_cond}        │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
   STAGE 4    │  Diffusion Policy          │   obs_cond' = [obs_cond ; r_t]
              │  1D-UNet + FiLM (or DiT)   │   → FiLM into every residual block
              └─────────────┬──────────────┘
                            ▼  action chunk A_t ∈ R^{Ta×7}
```

The perception front-end is deliberately **not** spiking. Spiking perception is a
well-explored, weakly-performing area; spiking *memory* is the interesting part. Keeping the
encoder dense also means we can share it with the DP for free.

---

## 4. Stage 1 — turning a dense stream into spikes

Two ingredients, both load-bearing.

### 4.1 Novelty gating (what to write)

A robot that is holding still should not be flooding its memory. Write bandwidth should track
*surprise*, not wall-clock. Maintain a cheap one-step predictor and spike on the residual:

```
ẑ_t   = (1 − λ) ẑ_{t−1} + λ z_{t−1}          # low-pass prediction, λ ≈ 0.3
e_t   = z_t − ẑ_t                             # predictive residual
g_t   = σ( (‖e_t‖ − μ_t) / s_t )              # scalar novelty gate ∈ (0,1), running-normalised
```

`g_t` scales the input drive into the bank. This is the same predictive-coding residual the
team's existing `RoboMemArena/predictive_coding_head` computes for the VLM — the SNN reuses that
signal as its **write-enable line**. Biologically this is novelty-gated encoding
(ACh/dopamine-gated hippocampal plasticity); computationally it stops long-τ neurons from
saturating during idle segments, which is their main failure mode.

### 4.2 Sparse distributed lift (how to write)

```
u_t = W_enc · (e_t ⊙ g_t),     W_enc ∈ R^{N_in×d}, sparse random, fixed
x_t = kWTA_k( u_t )            # exactly k of N_in units fire;  N_in = 2048, k = 64  (3 % density)
```

A fixed sparse random projection + k-WTA is a locality-sensitive hash: similar observations
produce overlapping codes, dissimilar ones near-orthogonal codes. This is the FlyHash / Sparse
Distributed Memory construction, and it is precisely the input format associative memory and
STDP want — dense codes make every STDP update interfere with every other.

*Ablation to run:* fixed-random `W_enc` vs. STDP-learned `W_enc`. My prior is fixed-random is
within noise and much simpler.

### 4.3 One clock, not two

**Decision: 1 SNN timestep = 1 control timestep (dt = 100 ms at 10 Hz).** No sub-stepping, no
Poisson rate coding over micro-time.

Rationale: the memory content we care about lives at 1–1000 control steps, not at millisecond
rate-code resolution. With `k`-WTA sparse codes a single binary step already carries
`log₂ C(2048,64) ≈ 400` bits — bandwidth is not the bottleneck. And one-clock makes the whole
bank an elementwise recurrence: fully batched on GPU, near-free compute.

The consequence is worth stating explicitly because it is the theoretical hook:

> With one clock and heterogeneous decay, the bank **is** a diagonal state-space model
> (à la S4/Mamba/LMU) with a binary threshold nonlinearity and a Hebbian-learned transition
> matrix. The τ-spread is the SSM's per-channel decay spectrum, arrived at from neuroscience
> instead of from HiPPO.

A multi-step variant (T_sub = 8, latency coding) stays in the codebase behind a flag for
neuromorphic-hardware deployment, where it is required.

---

## 5. Stage 2 — the bank

### 5.1 Neuron model

Per sub-bank `m` with membrane decay `β_m` and adaptation decay `ρ_m`, reset-by-subtraction ALIF:

```
V^m_t  = β_m ⊙ V^m_{t−1} + W^in_m x_t + W^rec_m s^m_{t−1} − θ^m_{t−1} ⊙ s^m_{t−1}
a^m_t  = ρ_m ⊙ a^m_{t−1} + s^m_{t−1}
θ^m_t  = θ_0 + γ a^m_t
s^m_t  = 1[ V^m_t > θ^m_t ]              then kWTA over the sub-bank
```

`β_m = exp(−dt/τ_m)`. Sub-banks (dt = 0.1 s):

| bank | τ_mem | β | horizon | role |
|------|-------|---|---------|------|
| B0 | 0.1 s | 0.37 | ~1 step | current percept, edge detector |
| B1 | 0.5 s | 0.82 | ~5 steps | motion / short phase |
| B2 | 2 s | 0.95 | ~20 steps | subtask phase |
| B3 | 10 s | 0.99 | ~100 steps | within-stage context |
| B4 | 60 s | 0.998 | ~600 steps | episode-level context |

Within a sub-bank, draw τ **log-uniformly around** the nominal value rather than fixing it —
heterogeneity within a population measurably improves robustness and memory span, and it costs
nothing (τ is a per-neuron vector). Optionally make `log τ` a learnable parameter for the
gradient-trained variant only.

### 5.2 Three memory substrates, three timescales

This is the intellectual core and should be the paper's figure 1:

| substrate | variable | timescale | memory type | plasticity |
|-----------|----------|-----------|-------------|------------|
| membrane potential | `V` | 0.1–1 s | **iconic** — what I am seeing now | none (dynamics) |
| adaptation / short-term plasticity | `a` (and Tsodyks–Markram `u,x` if enabled) | 1–60 s | **working** — what happened this episode | activity-driven, decays |
| synaptic weights | `W^rec` | episode → dataset | **semantic** — what usually follows what | **STDP** |

The user's brief was "different decay coefficients for long/short-term memory". The τ-spread
gives a *continuum inside the working-memory band*; the STDP weights give the long-term band for
free, and they are qualitatively different (content-addressable, non-decaying, capacity-limited
rather than time-limited). Saying this cleanly is most of the paper's contribution.

### 5.3 What STDP actually learns here

Not "features" — **transitions**. On recurrent connections with a causal window, STDP strengthens
`i→j` when `i` reliably fires shortly before `j`. Applied to a stream of sparse observation
codes, the bank becomes an **associative sequence predictor**: presenting the code for "drawer
open" partially pre-activates the codes that historically followed it. That is what makes the
readout *memory-like* rather than just *low-pass-filter-like*.

Trace-based pair STDP with soft bounds and a depression term:

```
p_t = α_pre  p_{t−1} + s_{t−1}                       # presynaptic eligibility trace
q_t = α_post q_{t−1} + s_t                           # postsynaptic trace
ΔW_ij = η [ (w_max − W_ij)·p_j s_i  −  A₋ · W_ij · q_i s_j ]
W ← W / ‖W‖₁,row                                     # synaptic scaling, per postsynaptic neuron
```

At dt = 100 ms, `α_pre` should span 2–5 steps for the fast banks and 10–50 for the slow ones —
i.e. **the STDP window scales with the bank's τ**. A single global STDP window across banks with
1000× different dynamics is a bug, not a simplification.

Three stabilisers are mandatory (every STDP paper that works has all three; every one that fails
is missing one):
1. **k-WTA lateral inhibition** — forces neurons to specialise on different patterns.
2. **Homeostasis** — per-neuron threshold adapts toward a target rate (~2–5 % duty cycle); kills
   dead and dominating units.
3. **Weight normalisation** — L1 per postsynaptic row; prevents runaway potentiation.

### 5.4 Optional third factor

A neuromodulated variant gates plasticity by a global scalar `M_t` (negative diffusion loss, or
a TD-style advantage): `ΔW ← M_t · ΔW`, with eligibility traces held over ~1 s. This makes the
bank task-aware without a single backward pass through time. Keep it as **phase 3**, not phase 1
— it adds a coupling between policy and memory that makes debugging much harder.

---

## 6. Stages 3–4 — reading out, and injecting into the Diffusion Policy

### 6.1 Readout

```
h_t = concat_m [ LP_{τ_m}(s^m_t) ;  a^m_t ]        ∈ R^{2·M·N}     (M=5, N=256 → 2560)
r_t = gφ(h_t) = LayerNorm(W₂ · GELU(W₁ h_t))       ∈ R^{d_cond}    d_cond = 128–256
```

`gφ` is trained by backprop **from the policy loss**, but `h_t` is treated as a constant
(`h_t.detach()`). This is standard reservoir-computing practice and it is what makes an LSM
work at all: the memory writes itself unsupervised, the policy learns *how to read* it. No
surrogate gradients, no BPTT, no gradient ever touches the spike nonlinearity.

### 6.2 Injection point

Diffusion Policy (Chi et al., RSS 2023) builds a global conditioning vector from the last
`To = 2` observations and FiLMs it into every residual block of the 1D conditional UNet. The
change is one line:

```python
# diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py
global_cond = torch.cat([obs_features.flatten(1), mem_r_t], dim=-1)   # + d_cond
```

For the transformer/DiT variant, append `r_t` as extra conditioning tokens for cross-attention —
strictly better, since the policy can attend to memory channels selectively. For the team's
π0.5 / MME-VLA path, `r_t` enters as extra prefix tokens.

**Do not feed past actions into the bank.** Only exteroceptive observations. Imitation learning
with action history reliably produces causal confusion / copycat behaviour, where the policy
learns to extrapolate its own previous action and ignores the scene. This is the single most
common way "adding memory" makes a manipulation policy *worse*, and it would be misread as the
SNN failing.

---

## 7. Training protocol, and the trick that makes it cheap

**Phase 1 — unsupervised write (no gradients at all).**
Stream every demo episode through the encoder + bank in temporal order; apply STDP online. One
to three passes over the dataset. Output: `W^rec`. Cost: negligible (elementwise recurrence).

**Phase 2 — cache the memory trace.**
With `W^rec` frozen, `r_t` is a deterministic causal function of the episode prefix. Run every
episode once and **cache `h_t` for every timestep to disk** (`T × 2560` floats per episode ≈
2.5 MB per 250-step episode at fp16). Now:

> Training a memory-augmented Diffusion Policy costs the same as training a memoryless one,
> for arbitrarily long history. No sequential batching, no BPTT, no truncated windows.

This is the practical payoff of refusing to backprop through the memory, and it is worth stating
as a contribution in its own right.

**Phase 3 — train the policy.** Standard DP denoising MSE; gradients flow through `gφ` and the
UNet only. Random-window sampling as usual, because `h_t` is already cached.

**Phase 4 (optional) — slow plasticity at deployment.** Keep STDP on at test time with a small
`η`. This is the claim-1 experiment: the memory keeps being written during evaluation.

*Practical note if you skip the cache:* the bank needs a warm-up. For a sampled index `t`,
replay from `t − W` (W ≈ 200 steps) with no gradient to prime `V, a`, then compute the loss at
`t`. Cheap, but the cache is strictly better.

---

## 8. Experiments

### 8.1 The go/no-go control (run this first, week 1)

**`SpikeBank(STDP)` vs. `SpikeBank(frozen random W^rec)`.**

Most of the memory in a spiking reservoir comes from the leaky dynamics, not from the learning
rule. If STDP does not beat a frozen-random recurrent matrix at equal size, the STDP story is
decoration and the project should be re-scoped to "heterogeneous-timescale spiking memory"
(still publishable, honest, and cheaper). Run this before building anything else.

### 8.2 Baselines on RoboMemArena (26 tasks, CSR / TSR)

| # | Method | What it isolates |
|---|--------|------------------|
| B1 | DP, `To=2` | official baseline |
| B2 | DP, frame-stack 16 | naive history |
| B3 | DP + GRU over history | learned recurrent memory |
| B4 | DP + Mamba/S4 over history | SOTA sequence memory, BPTT-trained |
| B5 | DP + frozen-random spiking reservoir | **dynamics without STDP** |
| B6 | DP + SpikeBank (ours) | full system |
| B7 | DP + oracle privileged state | ceiling |

### 8.3 Ablations

τ-spread (single τ vs. 5 banks) · number of banks · adaptive threshold on/off · novelty gating
on/off · k-WTA density · STDP window scaling with τ · detached vs. surrogate-gradient readout ·
one-clock vs. T_sub = 8.

### 8.4 Memory probing — direct tie-in with `memprobe`

Beyond task success, **decode memory content**: fit a linear probe from `r_t` to MemProbe-style
facts ("which object was placed in the basket at t = 40?") at increasing lag `Δt`. Produces a
*memory-retention curve per sub-bank* — the single most convincing figure available here, and it
shows directly whether the τ-hierarchy is doing what it claims. The MemProbe QA generator
already produces exactly these deterministic trajectory-derived facts.

---

## 9. Risks, honestly

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| STDP ≈ random reservoir | **High** | §8.1 first; fall back to a heterogeneous-τ-dynamics paper |
| Long-τ neurons saturate, lose selectivity | High | novelty gating + adaptive threshold + normalisation |
| Binary spikes bottleneck information | Medium | large N, sparse codes; graded-spike fallback |
| Causal confusion from history | Medium | exteroceptive-only memory; B2 baseline exposes it |
| "Why not just Mamba?" | **Certain** | answer is claims 1 & 2 (online gradient-free write, cache-cheap training), *not* raw success rate |
| STDP hyperparameters are fiddly | High | homeostasis makes it far less sensitive; sweep η, k, target-rate only |

---

## 10. Build order

1. `spikebank/neurons.py` — batched heterogeneous ALIF, one-clock. **Unit test: memory-retention curve per τ.**
2. `spikebank/encoder.py` — novelty gate + sparse lift + k-WTA.
3. `spikebank/stdp.py` — trace STDP, k-WTA, homeostasis, normalisation.
4. `spikebank/probe.py` — the §8.1 control + §8.4 retention curves on RoboMemArena HDF5, **before touching the Diffusion Policy**.
5. `spikebank/readout.py` + DP integration (one-line `global_cond` concat).
6. Full RoboMemArena eval sweep.

Steps 1–4 are ~2 weeks and answer the only question that matters. Step 5 is a day.

### Tooling

`snnTorch` or `Norse` for reference neuron implementations, but the one-clock ALIF is ~40 lines
of plain PyTorch and a hand-rolled version is easier to give per-neuron τ vectors and a custom
STDP rule. Recommendation: **hand-roll the bank in PyTorch**, use snnTorch only as a
cross-check. Lava/Loihi only matters if we make the neuromorphic-energy claim, which we should
not make in v1 without hardware numbers.
