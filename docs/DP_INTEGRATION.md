# Diffusion Policy integration — verified injection points, and one hard warning

*From a source-level read of `real-stanford/diffusion_policy` plus a 2026 sweep of
memory-conditioned policies. Verified against the released code unless marked.*

## 0. THE WARNING: do not build S4D-Real by accident

**This challenges the project's founding premise and must be resolved before Phase 1.**

The design says: *a bank of LIF neurons with different membrane decay constants gives long/short
term memory.* In state-space-model terms that is a diagonal SSM whose channels differ only in
`Re(A_n)`. **That is exactly the `S4D-Real` initialisation, and it is the variant S4D reports as
the weakest.**

In the winning recipes (`S4D-Lin`, `S4D-Inv`; Gu, Gupta, Goel, Ré, NeurIPS 2022,
[arXiv:2206.11893](https://arxiv.org/abs/2206.11893)) the decay is **fixed at `Re(A_n) = −1/2` for
every channel**. Diversity comes from two other places:

| source of diversity | S4D/S5/Mamba | our current design |
|---|---|---|
| **`Δ` — per-channel timestep**, log-uniform in `[10⁻³, 10⁻¹]` | **the main timescale knob** | absent |
| **`Im(A_n)` — oscillation frequency**, spread out (linear or inverse spacing) | **the main diversity knob** | absent (pure LIF has no resonance) |
| `Re(A_n)` — decay rate | held constant | **our only knob** |

Paper's own words: *"the real part of `A_n` controls the decay rate… `A_n = −1/2` is a good default
that bounds the basis functions by the envelope `e^{−t/2}`, giving a constant timescale"*, while the
imaginary parts give *"oscillating frequencies"* which *"should be spread out."*

**This independently predicts the E1 result.** We measured that the τ-spread composes at short lag
but dies by ~2 s, with the slow banks *not* outperforming the mid banks — which is what a
decay-only bank is expected to do.

**The spiking counterpart of the winning recipe is a resonate-and-fire population, not a LIF
population.** `ż = (b + iω)z + I(t)`, spike when `Im(z)` crosses threshold (Izhikevich 2001) **is
one diagonal-SSM channel plus a spike readout** — exactly, not loosely.

**Three responses, in increasing order of departure from the brief:**
1. **Add `Δ` heterogeneity.** Cheapest fix, keeps pure LIF: give each neuron its own effective
   timestep, log-uniform over three decades. This is the knob the SSM literature says carries the
   timescale diversity, and we currently do not have it.
2. **Add resonance.** Replace some sub-banks with **Balanced Resonate-and-Fire** neurons (Higuchi,
   Kairat, Bohté, Otte, **ICML 2024**, [arXiv:2402.14603](https://arxiv.org/abs/2402.14603)) — smooth
   reset via refractory damping preserves oscillation phase, stable over hundreds of steps.
3. **Adopt the explicit mapping.** **SiLIF** (Fabre, Dudchenko, Bouhadjar, Neftci,
   [arXiv:2506.06374](https://arxiv.org/abs/2506.06374)) writes AdLIF as a 2-state linear SSM
   `Ā = [[α, α−1], [a, β]]` with spike-triggered feedback, learns `τ_u, τ_w` and `Δt` in log space,
   and **C-SiLIF adopts complex diagonal S4D structure with S4D-Lin init**. This is the most direct
   SSM↔spiking-neuron correspondence published and it beats SSMs at half the compute.

**Recommended:** do (1) immediately — it is a one-line change to `neurons.py` and it is free. Run
(2) as the main ablation. Keep pure-LIF-with-τ-spread as the *baseline we are trying to beat*, not
as the proposal. Honest framing: *the brief's premise is the weakest point in the SSM design space,
and the fix is small.*

Related precedent worth knowing: **LMU memory cells can be implemented with `m` recurrently-connected
Poisson spiking neurons**, `O(m)` time and memory, error `O(d/√m)` (Voelker, Kajić, Eliasmith,
NeurIPS 2019) — "SSM in spikes" predates S4. Spiking follow-ups: **LMUFormer** (ICLR 2024,
[arXiv:2402.04882](https://arxiv.org/abs/2402.04882)), **L2MU** (ICANN 2024,
[arXiv:2407.04076](https://arxiv.org/abs/2407.04076), every element modelled with LIF populations),
NengoLoihi LMU on Loihi.

Also: `τ` spread should be **log-uniform, not uniform**. Tiling `[τ_min, τ_max]` needs
~`log(τ_max/τ_min)` log-spaced channels; uniform spacing wastes the bank on the fast end.
(`neurons.py` already does this via `het_octaves`; keep it.)

## 1. Verified injection point — UNet/FiLM variant

From `diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py`:

```python
global_cond_dim = obs_feature_dim * n_obs_steps      # line 135
this_nobs     = nobs[:, :To]                          # -> (B*To, ...)
nobs_features = obs_encoder(this_nobs)                # (B*To, Do)
global_cond   = nobs_features.reshape(B, -1)          # (B, To*Do)  <- ONE flat vector, no time axis
```

then in `conditional_unet1d.py`:

```python
global_feature = diffusion_step_encoder(k)                 # (B, 128)
global_feature = cat([global_feature, global_cond], -1)    # (B, 128 + To*Do) == cond_dim
```

That single vector goes to **every** `ConditionalResidualBlock1D` (all down/mid/up blocks):

```python
out   = self.blocks[0](x)                  # Conv1d -> GroupNorm(8) -> Mish
embed = self.cond_encoder(cond)            # Mish -> Linear(cond_dim, 2*C_out) -> (B, 2C, 1)
scale, bias = embed[:,0], embed[:,1]
out   = scale * out + bias                 # FiLM, per-channel, broadcast over time
```

Shapes: `x : (B, C, Tp)`, `cond : (B, cond_dim)`, FiLM broadcasts along `Tp`.
Defaults: `down_dims=[512,1024,2048]`, `kernel_size=5`, `n_groups=8`, `cond_predict_scale=True`.

**Our change is one line**, exactly as `DESIGN.md` §6.2 claimed:
```python
global_cond = torch.cat([nobs_features.reshape(B, -1), m_t], dim=-1)   # cond_dim += Dm
```

Two cautions the source makes obvious:
- **FiLM applies the same scale/bias at all `Tp` action positions** — memory cannot address
  specific future timesteps.
- `cond_encoder` is a single `Linear(cond_dim → 2·C_out)` per block, so **if `Dm ≫ To·Do` the memory
  dominates the FiLM statistics.** Normalise `m_t`, down-project, and **zero-initialise the output
  layer** (ControlNet-style) so training starts exactly at pretrained-DP behaviour.

**Verified DP defaults:** UNet `Tp=16, To=2, Ta=8`; Transformer `Tp=10, To=2, Ta=8`; DDPM 100 train
steps, 100 inference (10 DDIM for real-world); `squaredcos_cap_v2`, `prediction_type: epsilon`;
ResNet-18 (no pretraining) with **spatial-softmax pooling** and **GroupNorm instead of BatchNorm**,
separate encoder per camera; AdamW lr 1e-4, betas (0.95, 0.999), batch 64.

**EMA gotcha:** `EMAModel` averages module parameters. Any *stateful* part of the bank — membrane
potentials, adaptation, traces — **must be registered as a buffer, not a Parameter**, or EMA will
average the state. Our `neurons.py` uses buffers for state but `w_rec` is a `Parameter` with
`requires_grad=False`; check `EMAModel.step` does not average it, or move it to a buffer.

## 2. Verified injection point — transformer variant

From `transformer_for_diffusion.py`: `T_cond = 1 + n_obs_steps` (time token + `To` obs tokens);
`memory = encoder(cond_embeddings)`; decoder does causal self-attention over action tokens **then
cross-attention with `memory` as K/V**, with `memory_mask[t,s] = 0 if t >= s-1 else -inf`
(temporally aligned). Defaults `n_layer=8, n_head=4, n_emb=256, p_drop_attn=0.3`.

Here the observation axis is **preserved as separate tokens**, so memory enters as extra tokens:
`T_cond = 1 + To + Nm`, extend `cond_pos_emb`, zero-extend `memory_mask` so memory is always
visible. **Strictly more expressive than FiLM** — different denoising positions can attend to
memory differently. If the bank is grouped by timescale, **tokenise one token per τ-octave** so
attention can select the relevant timescale.

## 3. "History hurts" — the 2019–2022 consensus has been revised

The classic results still stand as mechanisms: **Causal Confusion in Imitation Learning**
(de Haan, Jayaraman, Levine, NeurIPS 2019, [arXiv:1905.11979](https://arxiv.org/abs/1905.11979)) and
**Fighting Copycat Agents** (Wen, Lin, Darrell, Jayaraman, Gao, NeurIPS 2020,
[arXiv:2010.14876](https://arxiv.org/abs/2010.14876)) — with history, `a_{t−1}` is recoverable from
`(o_{t−1}, o_t)` and the policy degenerates to `a_t ≈ a_{t−1}`.

**But two 2025–26 papers revise this for chunked diffusion policies:**

- **PTP — Learning Long-Context Diffusion Policies via Past-Token Prediction** (Torne, Tang, Liu,
  Finn, [arXiv:2505.09561](https://arxiv.org/abs/2505.09561)) finds diffusion policies show the
  **opposite** of copycat: they *fail to capture* essential past–future action dependencies. Fix is
  an auxiliary loss predicting *past* action tokens. **3× long-context performance, 10× training
  speedup.** Their recipe also caches long-context embeddings — the same idea as our §7 cache.
- **Training and Evaluating Diffusion Policies with Long Context Lengths** (Agarwal, Wei, Kargin,
  … **Tedrake**, MIT, [arXiv:2606.16447](https://arxiv.org/abs/2606.16447)) — **read this first.**
  *"naively scaling context length is not as brittle as advertised in literature"*; they attribute
  prior pessimism to not controlling for dataset size and to architecture choice.
  **UNet + Cross-Attention wins; UNet + FiLM degrades as context grows, partly because the FiLM
  conditioning parameter count grows with context length; DiT shows catastrophic failures.**

**This is a genuine differentiator for us, and it should go in the paper.** Their FiLM criticism is
that conditioning width grows with history length. **A recurrent/spiking bank is fixed-width by
construction**, so their negative result on FiLM does not apply to it. A fixed-width spiking memory
into FiLM is a design point they did not ablate.

→ `DESIGN.md` §6.2's blanket "history hurts, don't feed past actions" needs softening to: *past
observations are fine and helpful; the copycat risk is specifically about past **actions**, and even
that is contested for chunked diffusion policies.* Keep the exteroceptive-only memory as the
conservative default and the `DP + frame-stack` baseline as the diagnostic.

## 4. Nearest neighbours — the papers to position against

- **TFP: Temporally Conditioned Memory-Fusion Policies** (Liang et al., RSS 2026 SemRob workshop,
  [arXiv:2607.08283](https://arxiv.org/abs/2607.08283)) — **the closest published design to ours.**
  Maintains an episode-local task-progress belief with **Liquid Time-Constant dynamics** — an
  input-modulated leaky integrator — and injects it into a flow-matching action decoder via
  **adaptive modulation (AdaLN, i.e. FiLM)**. Their mechanistic analysis reports write-gain near
  manipulation events is **~6× larger** than in stable phases, and hidden-state interventions show
  the belief *causally* modulates the generated chunk. LIBERO 96.9→98.75, MIKASA ShellGameTouch 75.0.
  **This is our novelty gate and our conditioning path, in a non-spiking model, already published.**
  Our remaining differentiators: the spiking substrate, **STDP-learned recurrence** (theirs is
  gradient-trained), and deploy-time plasticity.
- **DSSP** ([arXiv:2605.14598](https://arxiv.org/abs/2605.14598)) — 2-layer **Mamba** history encoder
  (hidden 512, SSM state 64) → context token `c_t`; **hierarchical prefix conditioning**
  `C_t = [c_t, z_{t−N+1..t}]` with AdaLN on action tokens only; **dynamics-aware auxiliary loss**
  `1 − cos(g_φ(c_t,a_t), sg(z_{t+1}))`, λ=0.05. RoboTwin 2.0 **62.30 % vs DP3 55.24 %**; long-horizon
  subset 64.06 vs 52.76; 44.3M params vs DP3's 264.4M. **Note DSSP does not use FiLM** — if we do, we
  are choosing the harder path per the MIT study; justify or switch to cross-attention.
- **μVLA** ([arXiv:2606.12497](https://arxiv.org/abs/2606.12497)) — a **controlled isolation study of
  recurrence**; copy their ablation grid (memory width `m`, TBPTT length `K`, cross-step gradient vs
  detached EMA). MIKASA-Robo 0.42 → **0.84**; LIBERO no regression.
- **PRISM** ([arXiv:2606.16178](https://arxiv.org/abs/2606.16178)) — gated attention explicitly
  motivated as *"reducing the spurious correlations between the history and current action
  prediction"*; scales memory to ~2 minutes; ships **ReMemBench**.
- Also: **VPWEM** (RA-L 2026, fixed-count episodic compressor on a diffusion policy),
  **MemoryVLA(++)**, **HAMLET** (ICLR 2026), **Chameleon**, **CAMP**, **Mamba Policy** (IROS 2025),
  **MaIL** (CoRL 2024), **SAM2Act+** (ICML 2025).

**Disambiguate three senses of "memory" when writing:** (a) *state memory* — POMDP belief over hidden
state (ours, μVLA, CAMP, RSSM); (b) *long-context conditioning* — all history visible, just expensive
(PTP, DSSP, MIT study); (c) *corpus retrieval* — no episodic hidden state (Behavior Retrieval, STRAP,
VINN). **We are (a), with (b) as the efficiency argument.**

## 5. Benchmarks

**RoboMemArena is ours** ([arXiv:2605.10921](https://arxiv.org/abs/2605.10921)): 26 tasks / 4
categories, **68.9 % of subtasks memory-dependent**, average **>1000 steps**, 5 paired real tasks.
That memory-dependence fraction and horizon are the strongest justification available for this
project — use them in the motivation.

Add as external validity:
- **MIKASA-Robo** ([arXiv:2502.10550](https://arxiv.org/abs/2502.10550), `pip install
  mikasa-robo-suite`) — 32 tasks isolating **object / spatial / sequential / capacity** memory.
  `ShellGame`, `RememberColor`, `TakeItBack`, `SeqOfColors`. GPU-parallel ManiSkill3, state mode
  available as an MDP oracle.
- **MIT Push-and-Return / Grasp-and-Return / Marshmallows**
  ([arXiv:2606.16447](https://arxiv.org/abs/2606.16447)) — **native diffusion-policy memory tasks**,
  4–80 frames sim / 92 hardware. Most directly usable for a DP-conditioning experiment.
- **LIBERO as a no-regression control only.** LIBERO-LONG is long-horizon in the *compositional*
  sense, not the hidden-state sense; the scene is essentially fully observable. Memory methods gain
  only ~1–2 pts there (HAMLET 95.6→97.7, TFP 96.9→98.75, μVLA 96.2). Do not cite LIBERO as evidence
  of memory ability.
- **Memory Maze** ([arXiv:2210.13383](https://arxiv.org/abs/2210.13383)) ships **offline probing** —
  decode ground-truth layout from the agent's recurrent state. **Copy this protocol** for the bank;
  it is the same idea as our MemProbe plan and gives us a precedent to cite.

**Sobering baseline note:** POPGym (ICLR 2023) and Memory Gym (JMLR 2025) both report **plain GRUs
beating transformers, linear transformers and DNCs** on memory-heavy RL. Include a GRU baseline and
expect it to be strong.

## 6. Implementation checklist

1. **Compute `m_t` once per control step, outside the denoising loop.** DP runs 10–100 denoising
   iterations per chunk; the recurrence is O(N)/step and must be cached, not recomputed per
   diffusion iteration.
2. **Normalise across timescales before FiLM.** Slow channels have variance ∝ `1/(1−|λ|²)`; without
   LRU-style `γ_j = √(1−|λ_j|²)` scaling, a concatenated fast+slow vector is dominated by the slow
   channels and the FiLM affine is badly conditioned.
3. **Zero-initialise the `m_t` output projection** so training starts at pretrained-DP behaviour.
4. **Keep the denoiser an ANN.** Spiking only the memory pays the SNN cost once per control step
   instead of once per denoising iteration, and leaves unmodified DP as a clean ablation.
5. **Register all bank state as buffers** so EMA does not average it.
