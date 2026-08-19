# Prior art and positioning

*Compiled 2026-08-18. Verified by literature sweep; `[UNVERIFIED]` = search-snippet only.*

## 1. The name is taken — but the architecture is the inverse

**"Spiking Diffusion Policy" exists.**

- **SDP — Spiking Diffusion Policy for Robotic Manipulation with Learnable Channel-Wise Membrane
  Thresholds**, Hou, Gao, Yu, Yang, Ieong, 2024, [arXiv:2409.11195](https://arxiv.org/abs/2409.11195).
  Replaces the DP U-Net denoiser with a **fully spiking** U-Net (T_S=4). Push-T, BlockPush,
  Robomimic Lift/Can/Square/ToolHang/Transport. Claims 94.3 % dynamic-energy reduction (45 nm
  analytic SOP estimate, *not* measured).
- **STMDP — Spiking Transformer Diffusion Policy**, Wang, Sun, Lu, Zhang, Zeng (CASIA/BrainCog),
  2024, [arXiv:2411.09953](https://arxiv.org/abs/2411.09953). Spiking Transformer encoder + a
  Spiking Modulate Decoder. +8 % on Can.
- **L-SDPPO**, Zhang et al., 2026, [arXiv:2606.06049](https://arxiv.org/abs/2606.06049). SDP + DPPO
  RL fine-tuning, adds *state-dependent latency injection* — the closest existing "temporal
  dynamics as a functional mechanism in a spiking DP". **This line is live in 2026.**
- Upstream generative work: SDDPM (WACV 2024, [arXiv:2306.17046](https://arxiv.org/abs/2306.17046)),
  FSDDIM ([arXiv:2312.01742](https://arxiv.org/abs/2312.01742)), Spiking-Diffusion
  ([arXiv:2308.10187](https://arxiv.org/abs/2308.10187)).

**All of them spike the denoiser and pitch energy.** We keep the denoiser dense and put the SNN
*beside* the policy as memory, pitching **capability**. That inversion has no direct prior art —
and it dodges the energy-benchmark fight, where the honest measured numbers are unglamorous
(see §4).

**Naming consequence:** do **not** call this "Spiking Diffusion Policy". Use *SpikeBank*, or
"spiking memory conditioning for diffusion policies".

## 2. Nearest neighbour — read this one carefully

**Spiking Decision Transformers: Local Plasticity, Phase-Coding, and Dendritic Routing**,
Pandey & Biswas, Aug 2025, [arXiv:2508.21505](https://arxiv.org/abs/2508.21505).
LIF inside every attention block + **three-factor local plasticity** + phase-coded positional
encodings. Only CartPole/MountainCar/Acrobot/Pendulum; plasticity is auxiliary decoration, not
the memory mechanism; no manipulation, no diffusion. **Reviewers will cite this at us** — the
distinguishing claim must be that plasticity *is* the memory, on real manipulation.

Also: **Synaptic Motor Adaptation** (Schmidgall & Hays, ICONS 2023,
[arXiv:2306.01906](https://arxiv.org/pdf/2306.01906)) — three-factor rule doing something
functional in a robot, but online *adaptation*, not memory-for-conditioning.

## 3. The real baselines are ANN memory policies, not vanilla DP

Beating `DP(To=2)` proves nothing. The 2025–26 competition:

| Work | Link | Mechanism |
|---|---|---|
| **PTP — Long-Context Diffusion Policies via Past-Token Prediction** | 2025 | auxiliary past-token prediction; **~+50 % avg** |
| **VQ-Memory** (non-Markovian long-horizon) | [arXiv:2603.09513](https://arxiv.org/html/2603.09513) | VQ-VAE discretises past proprioceptive states into memory tokens |
| **DSSP — Diffusion State Space Policy, full-history encoding** | [arXiv:2605.14598](https://arxiv.org/html/2605.14598v1) | SSM as belief-state sufficient statistic |
| **MemoryVAM** | [arXiv:2606.20679](https://arxiv.org/html/2606.20679) | memory tokens into video backbone + action decoder |
| **Memory-gated diffusion policy** | Knowl.-Based Syst. 2025 | gated history |
| **Training/Evaluating DPs with Long Context Lengths** | [arXiv:2606.16447](https://arxiv.org/html/2606.16447) | |

They share our motivation — *long-horizon manipulation needs history to resolve perceptual
aliasing.* **DSSP is the head-to-head baseline** (§8.2 B4): SSM history encoding vs. our spiking
bank is the cleanest comparison, since both are diagonal-decay memories differing mainly in how
the recurrence is learned.

## 4. Do not lead with energy

Measured, end-to-end neuromorphic numbers are far below the analytic ones:

- **Loihi 2 / Astrobee free-flyer**, Stewart et al. (NRL + Intel), Dec 2025,
  [arXiv:2512.03911](https://arxiv.org/html/2512.03911): **0.217 J → 0.013 J per inference
  (~17×)**, latency 4.94 ms → 4.26 ms, with some control-precision loss. Sigma-Delta encoding,
  Lava-dl. **The most honest measured numbers available.**
- **SpikeVLA**, Song et al., June 2026, [arXiv:2606.27807](https://arxiv.org/html/2606.27807v1):
  first spiking VLA (navigation, not manipulation). GPU mem −62 %, energy 141 J → 49 J (**−66 %**).

So: 17× measured, −66 % on a real VLA, versus the 94–99 % claimed from 45 nm SOP counts.
**Claim capability, not joules.**

## 5. Heterogeneous timescales — established mechanism, unexplored application

- **Neural heterogeneity promotes robust learning**, Perez-Nieves, Leung, Dragotti, Goodman,
  *Nature Communications* 2021, [link](https://www.nature.com/articles/s41467-021-26022-3).
  Learned per-neuron τ improves accuracy and robustness; learned τ distributions match biology.
- **DH-SNN — temporal dendritic heterogeneity**, *Nature Communications* 2024,
  [link](https://www.nature.com/articles/s41467-023-44614-z). Per-dendritic-branch timing factors
  → multiple timescales inside one neuron. **Strongest existing implementation of the mechanism.**
- **PLIF — learnable membrane time constant**, Fang, Yu, Chen, Masquelier, Huang, Tian, ICCV 2021,
  [arXiv:2007.05785](https://arxiv.org/pdf/2007.05785). τ = 1/sigmoid(a), learned.
  ⚠️ SpikingJelly's `ParametricLIFNode` learns a **layer scalar**, not per-neuron.
- **A Heterogeneous SNN for Unsupervised Learning of Spatiotemporal Patterns**, 2021,
  [PMC7841292](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7841292/) — **STDP + heterogeneous τ,
  unsupervised.** Our mechanism minus the robot. Read closely.

"Heterogeneous τ helps SNNs" is **settled** — it cannot be the contribution. The open claim is
narrower and testable: *the τ-spectrum is a natural inductive bias for the multi-timescale
structure of long-horizon manipulation* (sub-second contact events ↔ minute-scale task phase).

## 6. Hybrid ANN↔SNN interface recipes (what to actually implement)

1. **Population encode / linear rate decode — PopSAN**, Tang, Kumar, Yoo, Michmizos, CoRL 2020,
   [arXiv:2010.09635](https://arxiv.org/abs/2010.09635). Learnable Gaussian receptive fields
   `A_E = exp(-½((s-µ)/σ)²)`, deterministic accumulate-and-fire; decode `fr = sc/T`,
   `a = W_d·fr + b_d`. Deployed on Loihi; ~140× energy vs Jetson TX2. **The canonical recipe.**
2. **Membrane-potential readout of non-spiking output neurons — DSQN**,
   [arXiv:2201.07211](https://arxiv.org/abs/2201.07211). `max_t v(t)` beats `mean_t v(t)`.
   Zero rate-coding quantisation loss.
3. **Low-pass filtered spike trains (LSM readout)**: `x_i(t) = α x_i(t-1) + s_i(t)`,
   `α = exp(-Δt/τ_read)`; train only the dense readout. **This is exactly our Stage-3 interface**,
   and a per-neuron `τ_read` is a *second, independent* heterogeneity axis on the decoder side.
4. **Sigma-Delta / delta modulation** (Loihi 2 native, Stewart et al. 2025) — our novelty gate is
   a close cousin; cite it as the hardware-native analogue.
5. **Bit-plane coding with a surrogate gradient**,
   [arXiv:2509.24411](https://arxiv.org/abs/2509.24411) — first surrogate gradient across a
   float→spike boundary, if we ever want gradients through the encoder.
6. **Learned direct/analog encoding** (no Poisson): cuts required latency from ~100 steps to 5–10.

## 7. Tooling

**SpikingJelly primary** ([Science Advances 2023](https://www.science.org/doi/10.1126/sciadv.adi1480)):
has a mature `STDPLearner` (trace-based, weight-dependent `f_pre`/`f_post`) that writes `-Δw` into
`.grad`, so **STDP layers and backprop layers can run under two separate optimizers in one loop** —
exactly our STDP-memory / gradient-policy split. Fastest measured backend (CuPy).

**snnTorch as escape hatch**: `snn.Leaky(beta=torch.rand(N), learn_beta=True)` gives per-neuron
decay in one line — SpikingJelly needs a `BaseNode` subclass for that.

Gotchas that will cost days if ignored:
1. Per-neuron τ is not free in SpikingJelly (layer-scalar by default). Don't lose a week; switch.
2. **Detach the SNN→diffusion boundary** or activation memory goes O(T) on top of the U-Net graph.
3. Use plain **SGD(lr=1, no momentum, no weight decay)** for STDP params — Adam distorts the rule.
4. `functional.reset_net()` **and** `STDPLearner.reset()` every batch or state leaks across episodes.
5. CuPy backend can't run on CPU — keep a `backend='torch'` path for CI.
6. Cost stacks multiplicatively: DDIM steps × SNN timesteps. SDP holds T_S=4 for this reason.
   (Our one-clock design avoids this entirely — a real advantage worth stating.)
7. **NIR** ([arXiv:2311.14641](https://arxiv.org/pdf/2311.14641)) as portable IR, adopt early.

*Our own bank is ~150 lines of plain PyTorch (see `spikebank/`), which sidesteps gotchas 1 and 5.*

## 8. Verdict on novelty

**Occupied:** spiking diffusion *denoisers*; spiking transformers/LMs/VLA; SNN locomotion +
Loihi energy numbers; "heterogeneous τ helps"; memory-conditioned diffusion policies (all-ANN).

**Open, and defensible:**
1. **Architectural inversion** — SNN *beside* a dense policy as temporal memory whose readout is
   conditioning, rather than inside the compute path for energy. No direct prior art.
2. **STDP as the memory-formation rule in a modern robot-learning stack.** In robotics STDP appears
   only as shallow feature extractors, LSM readout tuning, CPG tuning, or online adaptation.
   Nobody uses unsupervised STDP to build a persistent memory that a learned generative policy reads.
3. **No BPTT over the horizon.** This is the mechanistic argument: memory forms from demonstration
   data without the 1000-step credit-assignment problem that forces PTP and VQ-Memory into
   architectural workarounds. It also enables the cache trick (DESIGN.md §7).

**Threats, be ready:**
- "Why STDP and not BPTT?" → must show the ablation *same bank trained by BPTT/surrogate gradient*,
  or the claim collapses to "we used a worse learning rule".
- The energy card is already played; do not pick that fight.
- The SDP line is active — assume someone is months behind us on any spiking-DP variant. The
  **memory-bank framing is the moat, not the spiking-diffusion pairing.**
