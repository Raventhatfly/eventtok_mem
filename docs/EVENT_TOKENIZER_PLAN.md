# Event Tokenizer — research plan

**Status:** plan v0.1 — 2026-08-20
**Target:** CoRL workshop, ~4 weeks
**Data on disk:** `/n/netscratch/hankyang_lab/Lab/felix/dataset/robomme` and `.../robomemarena`

> **Project pivot.** This repo began as an STDP-trained spiking memory bank (see `DESIGN.md`,
> `EXPERIMENTS.md`). That line is parked. Three mechanisms survive the pivot and appear below:
> the **novelty gate** (now an event boundary detector), **FSQ** (now the tokenizer's quantizer),
> and **nested-dropout ordering** (now coarse-to-fine event tokens). The decay/accumulation
> machinery does not.

---

## 1. The claim

Tokenize a robot's own experience into a short sequence of discrete **event tokens**, feed them
back as context, and the policy has memory.

Two things make it a paper rather than an engineering exercise:

1. **Token sequences preserve repetition; consolidation-based memory destroys it.** Every
   token-bank method (MemoryVLA, VPWEM, PRISM, SAM2Act+) merges redundant entries, and repetition
   is exactly what merging discards. Ask a captioner to summarise three scoops and it says
   "scooping." A token sequence says `scoop scoop scoop`. **Compression is fine; deduplication is
   the killer.**
2. **Text-valued tokens transfer to every policy with zero adaptation.** Every modern VLA has a
   language channel, so the same memory string drops into π₀.₅, OpenVLA, RDT, GR00T with no
   adapter, no projection, no shared embedding space. You don't design a standardised token space —
   text already is one. This is the answer to the group's "reusable across WAM/VLA architectures"
   question, and the transfer experiment costs only inference.

---

## 2. What OAT gives us, and where we must diverge

`OAT: Ordered Action Tokenization` (Liu, Han, Gao, Zhao, Chen, Du — [arXiv:2602.04215](https://arxiv.org/abs/2602.04215))
is the template. Verified configuration and ablations:

| item | OAT value |
|---|---|
| input | **action chunks only** — no observations enter the tokenizer |
| action horizon `H_a` | 32 |
| latent tokens `H_l` | 8 → **4× compression** |
| latent dim `D_l` | 4 |
| FSQ levels | `[8,5,5,5]`, `|V| ≈ 1000` |
| encoder | 2-layer transformer, `d_model=256`, `d_head=64`, + learnable register tokens |
| decoder | 4-layer transformer, single-pass, self- + cross-attention |
| loss | reconstruction only: `‖â₁:Hₐ − a₁:Hₐ‖²` |
| optimiser | AdamW, lr 5e-5 (tokenizer/policy), 1e-5 (obs encoder) |
| ordering | nested dropout on registers + causal attention among registers |
| scope | trained **per-dataset** |

**Codebook-size ablation (LIBERO success rate) — read this carefully:**

| FSQ levels | \|V\| | success |
|---|---|---|
| `[8,6,5]` | 240 | **29.2** |
| `[8,8,8]` | 512 | 53.5 |
| `[8,5,5,5]` | **1000** | **56.3** |
| `[8,8,6,5]` | 1920 | 54.6 |
| `[7,5,5,5,5]` | 4375 | 46.9 |

**Nested dropout is not optional:** removing it costs 21 points on LIBERO (56.3 → 35.2), and
6–12 points on the other three benchmarks. Whatever else we change, keep it.

### The divergence that matters most

OAT compresses **4×** (32 action steps → 8 tokens) and its objective is exact action
reconstruction. For memory we need **50–100×**: a 1000-step episode should become 10–20 event
tokens, not 250.

You cannot reconstruct 100 steps of action from one token. **So OAT's reconstruction objective
will not survive the compression ratio we need.** This is the central technical problem of the
plan, and §4 is about it.

Second divergence: OAT tokenizes actions only. We need visual + action, because "the drawer was
already open" is not in the action trace.

---

## 2b. What is already on disk — verified 2026-08-20

`/n/netscratch/hankyang_lab/Lab/felix/dataset/robomme/`

**16 tasks x 100 episodes.** Task names map onto memory types:

| memory type | tasks |
|---|---|
| **counting** | `PickXtimes`, `SwingXtimes`, `BinFill` |
| order / sequential | `PatternLock`, `VideoPlaceOrder` |
| occlusion / reference | `ButtonUnmask`, `ButtonUnmaskSwap`, `VideoUnmask`, `VideoUnmaskSwap`, `PickHighlight`, `VideoPlaceButton` |
| other | `MoveCube`, `StopCube`, `InsertPeg`, `RouteStick`, `VideoRepick` |

`robomme_preprocessed_data/data/*.pkl` — one record per timestep:

```
image (256,256,3) · wrist_image (256,256,3) · state (8,) · actions (20,8)
prompt              "put two red cubes into the bin, then press the button to stop"
simple_subgoal      "pick up the first red cube"
grounded_subgoal    "pick up the first red cube at <105, 118>"
epis_idx · step_idx · exec_start_idx · is_demo
```

### The finding that changes the plan: **event boundaries and labels are already annotated**

`simple_subgoal` is a per-timestep string that changes exactly at event transitions. Verified on
episode 235 (472 steps, prompt *"first press both buttons on the table, then pick up the container
hiding the red cube, finally pick up another container hiding the green cube"*):

```
t=  0   press the first button
t=117   press the second button
t=213   pick up the container that hides the red cube
t=322   put down the container
t=363   pick up the container that hides the green cube
```

Five clean events with text labels, for free, on every episode of all 16 tasks.

**Consequences:**
1. **Segmentation ground truth is free** — better than the gripper heuristics of §3 tier 1.
2. **The text labels of §4.4 already exist.** No VLM captioning pass, no manual naming.
3. **Counting ground truth is free** — the ordinal is in the string ("the *first* red cube",
   "the *second*"), and the prompt states the target count.
4. The riskiest part of this project now has supervision available. **Take it.**

### Also on disk

- `robomme_data_h5/memory/siglip_moviechat_mme_v1/record_dataset_<TASK>/episode_*.npz` — 310 GB of
  **precomputed SigLIP features**. No encoder forward pass needed.
- `robomme_preprocessed_data/features/episode_*/` — `token_emb_*.npy` plus **`kept_indices.json`**.
  That file records which visual tokens survived MovieChat-style merging: **the dedup baseline's
  own artifact, already computed.** Straight into E4 as the straw-man.
- Raw `record_dataset_<TASK>.h5.tar.xz` (~4 GB each) if full frames are needed; decompress with the
  repo's `tarxz_h5.py`.
- RoboMME ships reference implementations for both **Diffusion Policy** (lerobot format) and
  **MemoryVLA** (TFDS) on this exact data — the policy baseline and the dedup baseline both exist.

### A negative result we already have

`/n/netscratch/ydu_lab/Lab/wfy/ckpts/robomme_policy_ckpt/` holds
`evaluation_moviechat_qformer_context_full` and `_requeue`. **MovieChat QFormer context was run
here as a baseline and its rollouts were poor.** It is not our method and we have never adopted
token merging.

That is useful, and it is motivation rather than self-critique: an off-the-shelf memory built on
consolidating similar visual tokens performs badly on this benchmark, in our own hands.

**But do not yet claim merging is the cause.** A bad rollout has many possible explanations —
tuning, the QFormer bottleneck, the training recipe — and attributing it to consolidation without
evidence is exactly the kind of claim a reviewer will take apart. The cheap test is the **per-task
breakdown** from those existing runs:

- If `PickXtimes` / `SwingXtimes` / `BinFill` fail *worse than* the non-counting tasks, that is
  causal evidence for the merging argument and it is the paper's opening figure.
- If it fails roughly uniformly across all 16 tasks, MovieChat is simply ill-suited here and the
  merging argument needs a controlled experiment instead: same policy, same data, memory with and
  without a dedup step.

Pull that breakdown before writing any motivation text.

### Positioning caution

MME-VLA already trains a **VLM subgoal predictor** on these annotations. So "produce subgoal text"
is occupied. The distinction to state explicitly:

> Subgoals are **prescriptive** — what to do next. Event tokens are **descriptive and historical** —
> what has already happened, in what order, how many times. One is a plan; the other is a record.
> The policy needs the record to know *which* subgoal it is on.


---

## 3. Segmentation — three tiers, build all three

The claim is *independent of how boundaries are found*. So do not build one clever segmenter and
defend it; build three cheap ones and show the result holds for all. That is a stronger paper and
much less risk.

**Tier 0 — subgoal annotations (free, exact, and now the main path).** `simple_subgoal`
transitions give boundaries and labels with no work at all (see §2b). Use these as the primary
segmenter for the headline result and as ground truth for tiers 1–3. Using provided labels does
weaken a *discovery* claim — but the claim here is about **memory**, not about discovering events,
and at four weeks reliability wins. Tier 3 then becomes the ablation showing it can be done
without labels.

**Tier 1 — proprioception heuristics (free, deterministic).**
`gripper_states` toggles open→closed→open; each transition is a boundary. Add contact onset from
joint torques and end-effector velocity zero-crossings. No learning, no annotation. For counting
tasks this may be *sufficient* — "scoop twice" is two grasp-release cycles. Ship this first: it
validates the entire downstream pipeline while the learned tokenizer is still a question mark, and
it doubles as **ground truth** for evaluating tiers 2 and 3.

**Tier 2 — fixed-length chunks (the baseline that tells you whether any of this matters).**
Chop every N steps, tokenize each. This is literally what OAT does. If fixed chunks match learned
boundaries on the counting task, segmentation was unnecessary — **and that is a finding, not a
failure.** Note that with fixed chunks a repeated event appears as a repeated token *n-gram*
rather than a single token; counting subsequences is slightly harder but sidesteps segmentation
entirely.

**Tier 3 — prediction-error boundaries (the principled version).**
Train a next-latent predictor; boundaries go where error spikes. This is **Event Segmentation
Theory** — humans segment at prediction-error peaks (Zacks & Tversky 2001; Reynolds, Zacks &
Braver 2007 computational model; Franklin, Norman, Ranganath, Zacks & Gershman's SEM). It is
task-agnostic, which is what keeps the tokens transferable, and it is the surviving piece of the
old novelty-gate design. Validate against tier 1's boundaries. External option: **Generic Event
Boundary Detection** (Shou et al., ICCV 2021) supplies a task definition and metrics.

---

## 4. The tokenizer

### 4.1 Architecture (OAT, extended to observations)

```
segment (variable length T_s)
  ├─ visual: frozen encoder per frame → pooled/attended over the segment
  └─ action: raw action sub-trajectory
              │
    transformer encoder + R learnable registers   (2 layers, d=256)
    causal attention among registers
              │
    z_1..z_R  ∈ R^{D_l}
              │
         FSQ per register                          (levels [8,5,5,5] → |V|=1000)
              │
    nested dropout: keep 1..K, mask the tail       (K ~ p, the ordering mechanism)
              │
    event token(s) T_1..T_R  — coarse → fine
```

`R` is small. For memory, aim at **R = 1–2 tokens per event**, versus OAT's 8 per action chunk.

The ordering earns its place twice over: the **coarse prefix is the nameable event type** and
repeats reliably across instances (which is what counting needs), while the **fine suffix carries
the specifics** (which cup, where). It also gives graded forgetting without any decay machinery:
keep full-depth tokens for recent events, truncate old ones to their coarse prefix.

### 4.2 The objective — the open problem

Reconstruction will not work at 50–100× compression. Four candidates, to be decided in week 1–2:

| objective | form | risk |
|---|---|---|
| **coarse-state reconstruction** | predict segment **endpoints** (start/end ee-pose, gripper, object positions) rather than the trajectory | may collapse to "arm moved from A to B", ignoring what happened |
| **contrastive on event identity** | same-type segments close, different-type apart; positives from tier-1 boundaries (e.g. two grasp-release cycles) | needs a notion of "same type"; tier-1 labels give it cheaply |
| **predictive** | token must predict the *next* segment's coarse state | ties the code to task dynamics; good for transfer |
| **semantic distillation** | match a frozen semantic embedding (DINOv2, or a VLM caption embedding) of the segment | inherits the teacher's notion of similarity, which may be right |

**My prior:** contrastive-with-tier-1-positives first, because it directly optimises the property
we need — *two instances of the same event must get the same code* — rather than hoping it falls
out of reconstruction. Reconstruction objectives allocate codes to whatever varies most in the
input, which on robot video is arm pose, not event type.

### 4.3 Codebook size

OAT's sweet spot was 1000 with sharp degradation at 240. But OAT needed action-reconstruction
fidelity; we need event *identity*, a much weaker requirement, so the optimum will move down.
Sweep `|V| ∈ {64, 256, 1000}`. Small vocabularies are also what make the labels nameable.

### 4.4 Naming

Once, offline, per code — never per frame at inference. Inspect a handful of clips per codebook
entry, assign a label (by hand or with one VLM pass). A finite codebook makes this a bounded job,
and this is precisely what a continuous embedding cannot offer.

---

## 4b. Tokenizer design — settled by the quantizer sweep (2026-08-20)

### The finding that decides the objective

**Reconstruction-trained codes are not semantic.** FSQ's own Appendix A.3, on both quantizers:

> *"We found no evidence that a particular code represents a fixed visual concept in either
> quantizer… individual codes do not learn very abstract concepts. Instead it is the combination
> of codes decoder weights which determine the final RGB image."*

**BEiT-v2 Table 4** shows the tension is not incidental but a direct trade:

| VQ-KD decoder depth | recon loss | codebook usage | IN-1k linear probe |
|---|---|---|---|
| 1 layer | 0.164 | **100 %** | **78.5** |
| 3 layers | 0.145 | 95 % | 77.9 |
| 6 layers | **0.136** | 77 % | **63.0** |

Better reconstruction → lower usage → worse semantics.

And **LOVE** (Jiang, Liu, Eysenbach, Kolter, Finn, NeurIPS 2022,
[arXiv:2212.04590](https://arxiv.org/abs/2212.04590)) makes the same point about *boundaries*:
LOVE vs VTA reach statistically indistinguishable ELBO (2838±19 vs 2868±43) while boundary F1
differs **0.91 vs 0.82**. **The likelihood term does not identify the boundaries; the compression
term does.** Their objective is literally (number of segments) × (bits per segment code):

```
L_CL(θ) = n_s · H_{p_z*}[z]      minimised subject to   L_ELBO ≤ C     (dual gradient descent on λ)
```

Note the sign: LOVE **minimises** marginal code entropy. The VQ/MAGVIT-v2 entropy penalty
`E[H(q)] − H(E[q])` **maximises** it (it is `−I(z;token)`; the second term is a rate *maximiser*
that forces uniform code usage). For an event vocabulary over repetitive data with a naturally
skewed event prior, forcing uniformity is actively wrong — it splits the most frequent motion
across many codes. Keep the per-sample confidence term; temper or invert the uniformity term.

### Consequences — five concrete design decisions

1. **Never reconstruct pixels.** Decoder target = frozen **DINOv2 patch features** at `t+k`
   (UniVLA) and/or the **action chunk** (QueST). UniVLA's own ablation: task-centric feature
   target **88.7** vs Genie-style all-visual-change **82.3** vs task-irrelevant-only **56.5**.
2. **Quantize the transition, not the frame.** Encode `(features_t, features_{t+k}, a_{t:t+k})`
   and let the code carry only what *changed*. Give the decoder frame `t` for free. This is why
   Genie's 8 codes come out as left/right/jump/no-op.
3. **Small vocabulary, few tokens.** Every latent-action model with good same-event→same-code
   behaviour uses a *tiny* book: Genie **8**, LAPA **8⁴**, IGOR **32**, UniVLA **16×2**,
   Moto **128**, villa-X **32**. Not 1000 flat. Genie is explicit: *"We limit the vocabulary size…
   to permit human playability and further enforce controllability (we use |A|=8)."*
   Our nameability requirement points the same way.
4. **Add explicit nuisance suppression, or expect camera-motion codes.** LAPA is the documented
   failure — codes meaning *"slight downward camera movement."* Three verified mitigations:
   IGOR's **mismatched random crops** between encoder input and decoder target; UniVLA's frozen
   task-irrelevant codebook + language conditioning; villa-X's **proprioceptive** forward model
   for motions "subtle in pixel changes but critical for control" (rotation, gripper).
5. **The contrastive-with-subgoal-positives objective (§4.2) is the right call** and is now
   supported by five independent sources rather than a hunch. VQ-KD-style **cosine distillation
   over DINOv2 feature deltas** is the natural alternative, and I found no paper doing exactly it.

### Quantizer: FSQ, `[8,8,8] = 512` or `[8,5,5,5] = 1000`

Direct domain precedent: **QueST** ([arXiv:2407.15840](https://arxiv.org/abs/2407.15840), NeurIPS
2024) uses FSQ `[8,5,5,5]` over 32-step action chunks and beats VQ **89.8 vs 81.2** on LIBERO-90
multitask, **68.8 vs 62.5** few-shot. FlexLAM independently picked the same levels for latent
actions.

Caveat worth knowing: FSQ's own paper reports *"for low codebook sizes… VQ marginally outperforms
FSQ"*, with the crossover at **|C| = 2¹⁰**, i.e. right at our target. Three counter-data-points say
take FSQ anyway — QueST above; BSQ Table 5 where VQ at 1024 codes sat at **57.5 % usage / rFID
7.05**; and the rotation-trick paper where a K=1024 VQGAN fell to **27 % usage** on a low-diversity
dataset. FSQ's real value here is operational: no commitment loss, no EMA, no k-means init, no
dead-code replacement, ~100 % usage by construction. The research risk belongs in the objective,
not in quantizer babysitting.

Keep `L_i ≥ 5` (FSQ's own heuristic — below that "subpar performance"). **BSQ** at L=9–10 is the
alternative if we want an explicit rate knob: it is the only quantizer here with a tunable γ on the
marginal-entropy term, which is exactly the dial between "use every code" and "let frequent events
collide" — and collision is what counting needs.

### Adaptive length

Nested dropout over event tokens with a **power-of-2 keep-length schedule** (FlexTok's ablated
choice). Avoid the geometric schedule without Rippel's unit-sweeping fix — high-index units starve
(p ≈ 3e-5 for the 100th unit at ρ=0.9). For a stopping rule, prefer an **InfoTok-style one-pass
ELBO router** over ElasticTok's binary search: same effect, one decoder pass instead of log₂N, and
no reliance on the monotonicity assumption ElasticTok itself admits is false.

### Instrumentation — three pathologies

1. **FSQ dimension collapse.** Aggregate codebook usage is the wrong instrument for FSQ. Log the
   **per-channel histogram over each channel's L_i levels**: with d=3–4, one channel collapsing to
   1–2 levels silently costs a factor of 5–8 of effective vocabulary while total usage still looks
   healthy.
2. **Nuisance capture.** Measure `P(same code | same subgoal)` vs `P(same code | different
   subgoal)` — this is E3, and the subgoal labels give it for free. Also mutual information
   between code and episode id / camera / lighting / object instance, and a **repeat test**: edit
   distance between code sequences for N repetitions of the same demonstration.
3. **Uniformity fighting the event prior.** Plot the code histogram against the empirical
   subgoal-frequency histogram. Perfect uniformity over 512 codes when three motions are 80 % of
   the data is a failure, not a success.


---

## 4c. The pipeline — settled (2026-08-20)

### BPE over a code stream, not run-length encoding

The text-tokenizer analogy is not a metaphor; it is the implementation. **PRISE** (Zheng, Cheng,
Daumé III, Huang, Kolobov, ICML 2024, [arXiv:2402.10450](https://arxiv.org/abs/2402.10450)) is the
only published method emitting **variable-duration discrete tokens from LIBERO demos**, and it does
it in two stages:

1. per-timestep state-conditioned VQ with a **tiny alphabet (C = 10)**
2. **BPE over the resulting code stream** — vocab 200, `min_frequency 10`, `max_token_length 20`

Each BPE token is a tuple `⟨f_ξ, L_ξ⟩` — a skill with its own horizon. Variable length is purely
frequency-driven, exactly as in text.

BPE dominates the run-length-encoding idea: RLE only merges *identical adjacent* codes, whereas BPE
learns frequent multi-code *patterns*, so a "scoop" made of codes `A B C` becomes one token. And it
preserves counts for the same reason RLE does — merges are over adjacent pairs within an event,
never across separated occurrences.

**The reusable artifact:** PRISE's `tokenizer_api.py` is a **standalone ~110-line wrapper** around
HuggingFace `tokenizers` with no dependency on the rest of the repo. It maps integer code IDs onto
a ByteLevel alphabet, trains BPE/WordPiece/Unigram with `vocab_size` / `min_frequency` /
`max_token_length`, and exposes `encode(list_of_int_lists)` / `decode`. **Drop it on any per-step
code stream.** Stage II costs ~10 min on one A100.

Skip PRISE's Stage I — it needs 4× A100 40 GB + 400 GB RAM as published.

### Where the per-step codes come from

**QueST** ([arXiv:2407.15840](https://arxiv.org/abs/2407.15840), `github.com/pairlab/QueST`, MIT,
★116). Verified from source: causal strided conv (`strides [2,2,1]`, `kernels [5,3,3]`, F=4) +
2-layer transformer encoder (d=256) → **8 tokens per 32-step chunk**, FSQ `[8,5,5,5]` = 1000,
4-layer transformer decoder, L1 reconstruction only, no commitment loss, no EMA.

Two facts that matter:
- `config/task/libero_base.yaml` already declares **exactly our keys**: `agentview_rgb`,
  `eye_in_hand_rgb`, `joint_states`, `gripper_states`, `ee_pos`.
- `SkillVAE.encode(act, obs_emb=None)` **already has the visual hook**, but
  `load_obs_for_pretrain: false` — the released tokenizer is action-only. **Wiring `obs_emb`
  through `compute_autoencoder_loss` is the one piece of real engineering**, and it is precisely
  the visual+action tokenizer we want.

If codebook collapse bites, **STAR** (ICML 2025 spotlight, `iLearn-Lab/ICML25-STAR`) is the same
codebase with rotation-augmented residual quantization — 93.6 vs QueST 81.5 on LIBERO-90. Note the
paper's own `JiuTian-VL/STAR` link is dead.

### Boundary detectors that run today (for validation, not for the main path)

- **UVD** (Zhang et al., ICRA 2024, [arXiv:2310.08581](https://arxiv.org/abs/2310.08581),
  `github.com/zcczhang/UVD`) — **training-free**. Walk backward in a pretrained embedding space;
  distance-to-goal decreases within a subtask and jumps at boundaries. One-line API:
  `uvd.get_uvd_subgoals(video, preprocessor_name="vip")` on an `(L,H,W,3)` array. **Runs on
  `agentview_rgb` immediately.**
- **PerAct keyframe heuristic** — ~20 lines, the de-facto baseline:
  ```python
  small_delta = np.allclose(obs.joint_velocities, 0, atol=0.1)
  stopped = (stopped_buffer <= 0 and small_delta and not next_is_not_final
             and gripper_state_no_change)
  if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
      episode_keypoints.append(i)
  ```
  with `stopped_buffer = 4 if stopped else stopped_buffer - 1`. Needs only `joint_states` +
  `gripper_states`.
- **SBD** (Deng et al., **ICCV 2025**, [arXiv:2503.10684](https://arxiv.org/abs/2503.10684)) — the
  prediction-error boundary detector, explicitly motivated by human event-segmentation theory. Run
  an unconditional action predictor; boundary where per-step NLL spikes:
  `if loss − mean(loss_history) > GAP` with **GAP = 18**, running mean reset after each boundary,
  segment lengths pruned to [15, 200]. This is the published version of our novelty gate.

### The nearest neighbour — differentiate from this explicitly

**EventVLA — "Event-Driven Visual Evidence Memory for Long-Horizon VLA Policies"**
([arXiv:2606.20092](https://arxiv.org/abs/2606.20092), 2026). An MLP head on the VLA's hidden state
predicts future-keyframe probability over an H=50 horizon with raised-cosine soft labels; when
`p̂ ≥ 0.55` the **raw image** is committed to a FIFO event buffer (`N_max = 5`) after 1-D NMS, and
memory re-enters as `concat([A_t, E_{t−1}, o_t])`. **+40 % over SOTA memory-augmented VLAs.** No
code released.

They buffer **raw images**; we buffer **discrete tokens**. That single swap is the contribution,
and it is what gives counting, nameability, a bounded vocabulary, and zero-adaptation text
transfer. State the comparison in the intro rather than letting a reviewer find it.

### Two data gotchas nobody in this literature handles

1. **Absolute vs. relative state.** `ee_states` are absolute world-frame poses, so the *same* skill
   toward two object placements looks completely different. Transform to object-centric or
   start-relative frames before tokenizing, or codes will fragment by location.
2. **Dimensionality vs. data.** ~5–10k timesteps per task. Reduce to `[ee_pos(3), ee_ori(3),
   gripper(1)]` = 7 dims rather than the full 23-D state.

### Resulting build

```
features (precomputed SigLIP)  +  actions
        │
   QueST SkillVAE with obs_emb wired in     ← the one real engineering task
        │
   FSQ [8,5,5,5] → per-chunk code indices
        │
   PRISE tokenizer_api.py: BPE, vocab ~200  ← standalone, ~10 min
        │
   variable-length event tokens
        │
   name each token by its modal `simple_subgoal`
```

Validate boundaries against `simple_subgoal` transitions, UVD, and the PerAct heuristic.


---

## 4d. Novelty threats — checked 2026-08-20, and one is serious

### WeaveLA — read this before committing

**"WeaveLA: Event Driven Cross-Subtask Latent Memory Weaving for Repetitive Robot Manipulation"**,
Shoujing Zhu et al., 16 June 2026, [arXiv:2606.17463](https://arxiv.org/abs/2606.17463).

- **RoboMME benchmark.** Same one.
- **π₀.₅ backbone.** Same one.
- **"Repetitive Robot Manipulation."** Same problem.
- Detects **subtask completion** as the event trigger, compresses the finished segment into
  **latent tokens** via query-driven attention pooling, injects them into the next subtask's action
  generation. Frozen base policy, minimal overhead.
- **`SwingXtimes` (N=3): 0 % → 47.8 %.** One of our two target counting tasks.
- No degradation on single-execution episodes; gains appear only where cross-subtask information
  is required.

**This occupies "event-triggered memory for repetitive manipulation on RoboMME."** It was posted
two months ago. Treat the slot as taken and re-derive what is left.

### What is still ours

Three axes survive, and they are narrower than they were:

1. **Discrete vocabulary vs. continuous latents.** WeaveLA pools into continuous latents. Discrete
   codes buy nameability, counting by token occurrence, a bounded vocabulary, interpretability,
   and text transfer. This is the main remaining axis.
2. **Append-only log vs. one-step handoff.** WeaveLA *weaves* the previous subtask into the next.
   That is a handoff, not a record of everything that has happened. A token log accumulates.
3. **Zero-adaptation cross-architecture transfer.** WeaveLA's latents are π₀.₅-specific; text
   tokens are not.

### The wedge — and it is testable

**A latent handoff should not scale in N; a token log trivially does.** WeaveLA reports N=3. Ask
what happens at N=5, N=7, N=10. A fixed-size pooled latent has to encode "how many" in its
magnitude or geometry; an append-only token sequence just gets longer.

So the count-extrapolation experiment (E5) is promoted from "nice ablation" to **the central
experiment**. If we match WeaveLA at N=3 and beat it at N≥5, the paper stands on a mechanism
argument rather than a leaderboard delta.

Also worth noting: WeaveLA's memory-free baseline is **0 %** on `SwingXtimes`. That confirms the
motivation — counting tasks are genuinely broken — and it means the interesting comparison is
against WeaveLA, not against a memoryless policy.

### The other two, lower threat

**KEMO — "Event-Driven Keyframe Memory for Robot Manipulation"**, Yihan Zeng et al., 22 June 2026,
[arXiv:2606.23589](https://arxiv.org/abs/2606.23589). Combines robot kinematics with visual
filtering to detect events, encodes selected keyframes as "compact temporally ordered memory
tokens," fuses via cross-attention + gated residual. Real-world dual-arm, 830–2846-step
trajectories, **+23.6 % TSR / +34.1 % SCR** over memory-free. Ablations show event-driven selection
beats uniform sampling and recent-frame retention. **Does not address repetition or counts** — it
*selects* task-relevant keyframes, which is a filtering criterion, and filtering is where counts
die. Its tokens are keyframe embeddings, not a learned discrete vocabulary.

**OmniAct** ([arXiv:2606.27251](https://arxiv.org/abs/2606.27251), Shi, Huai, Wang et al., 25 June
2026). "Adaptive hierarchical memory with event-boundary-driven compression for sub-linear context
growth," near-flat token consumption over 100k+ accumulated interaction tokens, 40 real-world
long-horizon tasks with IoT coordination. Omnimodal agent scope rather than a manipulation policy;
mechanism not extractable from the abstract. **Pull the PDF before writing related work.**

### Revised novelty statement

> Boundary-coupled variable-length **quantization** feeding an **append-only, count-preserving**
> event token log, delivered as text so it transfers across policies without adaptation.

Every word in that sentence is doing work. Drop "discrete", "append-only", or "count-preserving"
and it collapses into WeaveLA or KEMO.


---

## 5. Experiments

**E1 — does the memory string change behaviour at all? (week 1, before building anything)**
Hand-write memory sentences ("you have already scooped twice") and prepend them to a VLA on a
counting task. If a *correct hand-written* memory does not change behaviour, the whole delivery
mechanism is dead and everything downstream is moot. This is the go/no-go and it costs an
afternoon.

**E2 — segmentation quality.** Tier 3 boundaries vs tier 1 ground truth: F1 with a tolerance
window. Tier 2 as the floor.

**E3 — code consistency (the property the paper rests on).** Do two instances of the same event
get the same code? Report within-type vs between-type code agreement. This replaces the old
sparse-overlap test and is the same question in discrete form.

**E4 — counting.** Probe the count from the token sequence. Baselines: memoryless, a
similarity-merged token bank (the dedup straw-man, and the point of the paper), a VLM captioner
(≈ MemER), and a GRU. Prediction: merging and captioning lose counts; token sequences don't.

**E5 — extrapolation. THE central experiment (see §4d).** Counts beyond the training range, and
beyond WeaveLA's reported N=3. A pooled latent must encode "how many" in its geometry; an
append-only token sequence just gets longer. Match at N=3, win at N≥5.

**E6 — zero-adaptation transfer.** Same memory string into ≥3 VLAs, inference only, no
fine-tuning. Then few-shot: briefly tune each to attend to the string. Expect few-shot > zero-shot;
the gap is a result.

---

## 6. Build order

1. **Week 1** — E1 first. Then tier-0 segmentation from `simple_subgoal` transitions on the
   preprocessed pkls, plus the readable-timeline figure. Start with `PickXtimes` and `SwingXtimes`,
   the two purpose-built counting tasks.
2. **Week 2** — FSQ tokenizer over segments, objective bake-off (§4.2), E3.
3. **Week 3** — E4/E5 counting and extrapolation; tier-2 and tier-3 segmenters as ablations.
4. **Week 4** — E6 transfer, then write.

Everything through week 3 is offline: cached features, forward passes, linear probes. **No
simulator, no rollouts, no policy training.** That is what makes four weeks feasible.

---

## 7. Risks

| risk | severity | mitigation |
|---|---|---|
| VLAs ignore the memory string | **fatal** | E1 in week 1; phrase as natural language, not token soup |
| codes don't collide for same-type events | **fatal** | contrastive objective targets this directly, with positives from subgoal labels; E3 measures it |
| reconstruction objective fails at 50–100× | high | §4.2 bake-off; don't inherit OAT's loss unexamined |
| prior art (VQ-Memory, QueST, MemER, Mimir) | high | see `PRIOR_ART.md`; differentiate on count preservation + text transfer |
| segmentation is unnecessary | medium | tier 2 exposes it; report it honestly as a finding |
| "you just used the provided subgoal labels" | high | tier 3 unsupervised ablation; and the claim is about memory, not event discovery |
| overlap with MME-VLA's subgoal predictor | medium | record vs. plan distinction (§2b); counts are what subgoal prediction does not give |
| "this is just prompting" | certain | the contribution is the tokenizer; text is delivery |

## 8. Open questions for the next meeting

- Is the tokenizer trained per-dataset (OAT's choice) or shared across datasets? Sharing is what
  the transfer story needs, and OAT does not test it.
- Does the coarse/fine split actually align with event-type vs instance-specifics, or is that
  wishful? E3 with per-prefix-depth code agreement answers it.
- Do we tokenize vision+action jointly, or two parallel vocabularies? Joint is simpler; separate
  lets us ask which modality carries the event identity.
