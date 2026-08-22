# Event Tokenizer — plan

**v1.0 — 2026-08-21** · target: CoRL workshop, ~4 weeks
*(Supersedes `_archive_event_tokenizer_v1.md`, which records how we got here. The earlier spiking-memory
line is parked; see `DESIGN.md`.)*

---

## 1. The claim

Tokenize a robot's own experience into a short sequence of discrete **event tokens**, feed them back
as context, and the policy has memory.

> **Novelty statement.** Boundary-coupled variable-length **quantization** feeding an
> **append-only, count-preserving** event token log, delivered as **text** so it transfers across
> policies without adaptation.

Every word is load-bearing. Drop "discrete", "append-only", or "count-preserving" and the idea
collapses into WeaveLA or KEMO (§7).

Two supporting arguments:

1. **Token sequences preserve repetition; consolidation destroys it.** Methods that merge redundant
   entries discard exactly the information counting needs. Ask a captioner to summarise three scoops
   and it says "scooping"; a token log says `scoop scoop scoop`. *Compression is fine — dedup is the
   killer.*
2. **Text-valued tokens transfer with zero adaptation.** Every modern VLA has a language channel, so
   the same memory string drops into π₀.₅, OpenVLA, RDT, GR00T with no adapter and no shared
   embedding space. The transfer experiment costs only inference.

---

## 2. What is already on disk

`/n/netscratch/hankyang_lab/Lab/felix/dataset/robomme/` — 16 tasks × 100 episodes.

| memory type | tasks |
|---|---|
| **counting** | `PickXtimes`, `SwingXtimes`, `BinFill` |
| order | `PatternLock`, `VideoPlaceOrder` |
| occlusion / reference | `ButtonUnmask(Swap)`, `VideoUnmask(Swap)`, `PickHighlight`, `VideoPlaceButton` |
| other | `MoveCube`, `StopCube`, `InsertPeg`, `RouteStick`, `VideoRepick` |

`robomme_preprocessed_data/data/*.pkl`, one record per timestep:
`image (256,256,3)`, `wrist_image`, `state (8,)`, `actions (20,8)`, `prompt`,
**`simple_subgoal`**, `grounded_subgoal`, `epis_idx`, `step_idx`.

**Event boundaries and labels are already annotated.** `simple_subgoal` changes exactly at event
transitions. Verified, episode 235 (472 steps):

```
t=  0   press the first button
t=117   press the second button
t=213   pick up the container that hides the red cube
t=322   put down the container
t=363   pick up the container that hides the green cube
```

Also available: **310 GB of precomputed SigLIP features**
(`robomme_data_h5/memory/siglip_moviechat_mme_v1/record_dataset_<TASK>/episode_*.npz`) — no encoder
forward pass needed. Raw `.h5.tar.xz` (~4 GB/task) if full frames are wanted.

**Use the subgoal labels for naming and evaluation only, never for training.** Keeping them out
keeps the codes task-agnostic, which is what the transfer claim rests on, and "we never used the
annotations" is a stronger result.

---

## 3. The tokenizer

### 3.1 Framing: inverse dynamics over a transition, both streams in and out

Video and actions are not redundant, and the difference is what memory must record:

| action | visual change | event |
|---|---|---|
| gripper closes | cube rises | successful grasp |
| gripper closes | nothing moves | **failed grasp** |
| no action | object moves | external event |

Including the action stream is also the fix for LAPA's documented failure mode — codes that mean
*"slight downward camera movement."* That happened because LAPA is vision-only; villa-X showed
proprioceptive grounding suppresses it.

### 3.2 Architecture

```
every k frames:
  encoder tokens:  R learnable registers  (R = 1–2)
                   SigLIP patch features at t          ← precomputed
                   SigLIP patch features at t+k
                   action chunk a_{t:t+k}, per-step linear embed  (k × 8)
                   + modality embedding per group
  transformer, 2–4 layers, d = 256, causal attention among registers
  → z_1..z_R  →  FSQ  [8,8,8] = 512   (or [8,5,5,5] = 1000)

  decoder, two heads:
     A:  (features_t , code) → predict features_{t+k}    cosine loss
     B:  (code)              → predict a_{t:t+k}          L1
```

**The dual head forces both modalities into the code.** Action-only codes fail head A (same action,
different outcomes); vision-only codes fail head B. Ablating each head afterwards reports what the
code retained.

Head A receives `features_t` for free, so the code encodes only *what changed*. This is why Genie's
8 codes come out as left/right/jump rather than scene descriptors.

### 3.3 Settled hyperparameters, and why

**Never reconstruct pixels.** Five independent results:

- FSQ Appendix A.3: *"We found no evidence that a particular code represents a fixed visual concept
  in either quantizer."*
- BEiT-v2 Table 4 makes it a measured trade — recon loss 0.164→0.136, codebook usage 100%→77%,
  linear probe **78.5→63.0**.
- LOVE (NeurIPS 2022): identical ELBO, boundary F1 **0.91 vs 0.82**. *The likelihood term does not
  find boundaries; the compression term does.*
- UniVLA: feature-space task-centric target **88.7** vs pixel-style all-visual-change **82.3** vs
  task-irrelevant-only 56.5.
- REPA: reconstruction *"is not capable of eliminating unnecessary details."*

**`k = 20` (1 s at 20 Hz).** Episode 235's events run 41–117 steps, so an event spans 2–6 codes —
enough repetition for BPE to find the pattern. A 472-step episode → 23 codes → ~5–8 event tokens,
matching its 5 annotated subgoals. Matches LAPA (~0.6 s) and UniVLA (~1 s). At `k=100` you get one
code per event and nothing to merge.

**Small vocabulary.** Every latent-action model with semantic codes uses a tiny book: Genie **8**,
LAPA 8⁴, IGOR **32**, UniVLA 16×2, Moto **128**. Genie is explicit: *"we limit the vocabulary size…
to permit human playability and further enforce controllability (we use |A|=8)."* Keep FSQ's
`L_i ≥ 5` heuristic.

**FSQ over VQ.** QueST runs `[8,5,5,5]`=1000 on robot action chunks and beats VQ **89.8 vs 81.2** on
LIBERO-90, **68.8 vs 62.5** few-shot. FSQ needs no commitment loss, no EMA, no k-means init, no
dead-code replacement. FSQ's own paper notes VQ marginally wins below |C|=2¹⁰, but VQ at 1024 codes
sat at 57.5 % usage in BSQ's Table 5 and fell to 27 % on a low-diversity dataset in the
rotation-trick paper. Take FSQ; put the research risk in the objective, not in quantizer babysitting.

### 3.4 Codebase

**QueST** (`github.com/pairlab/QueST`, MIT, ★116). Verified from source: causal strided conv
(`strides [2,2,1]`, `kernels [5,3,3]`, F=4) + 2-layer transformer (d=256) → 8 tokens per 32-step
chunk; FSQ; 4-layer transformer decoder; L1 reconstruction only.

Two facts: `config/task/libero_base.yaml` already declares `agentview_rgb`, `eye_in_hand_rgb`,
`joint_states`, `gripper_states`, `ee_pos`. And **`SkillVAE.encode(act, obs_emb=None)` already has
the visual hook — it is just disabled** (`load_obs_for_pretrain: false`). Wiring `obs_emb` through
`compute_autoencoder_loss` *is* the visual+action tokenizer, and it is the one piece of real
engineering.

If codebook collapse bites: **STAR** (ICML 2025, `iLearn-Lab/ICML25-STAR`) is the same codebase with
rotation-augmented residual quantization, 93.6 vs QueST 81.5 on LIBERO-90. The paper's own
`JiuTian-VL/STAR` link is dead.

---

## 4. From codes to event tokens: BPE

The text-tokenizer analogy is the implementation. **PRISE** (ICML 2024,
[arXiv:2402.10450](https://arxiv.org/abs/2402.10450)) is the only published method emitting
variable-duration discrete tokens from LIBERO demos: per-step VQ with a tiny alphabet, then **BPE
over the code stream** (vocab 200, `min_frequency 10`, `max_token_length 20`). Each BPE token is
`⟨skill, horizon⟩`; variable length falls out of merge frequency, as in text.

**Reusable artifact:** PRISE's `tokenizer_api.py` is a standalone ~110-line wrapper around
HuggingFace `tokenizers` with no dependency on the rest of the repo. Takes any list-of-int-lists,
exposes `vocab_size` / `min_frequency` / `max_token_length`. ~10 min on one A100. **Skip PRISE's
Stage I** — as published it wants 4× A100 40 GB + 400 GB RAM.

### A second trap: BPE is not streaming-stable

Text BPE is applied to a *complete* sequence. A robot's codes arrive one at a
time, and applying BPE to the growing prefix is not stable — with a merge
`(A,B) -> X`, the stream `A` tokenizes to `[A]` and one step later `A B`
tokenizes to `[X]`. The earlier token was not appended to, it was **replaced**.
A log built that way rewrites its own history and any count read from it can
change retroactively, which defeats the entire design.

**The fix is where BPE runs, not whether.** Confine merges to a single *completed*
event span:

```
codes:  A A A | B B | C C C C | B B ...
              ^ boundary -> span [A A A] is final -> BPE it -> append, immutable
```

Nothing after a boundary may merge into what precedes it, so the log is
append-only **at event granularity**. Latency is one transition. The two guards
above make this sound: no self-merges means repetitions can never collapse, and
boundaries as hard barriers during vocabulary training means no learned merge
ever spans one.

Implemented in `eventtok/bpe/streaming.py`; the append-only property is asserted
directly in `tests/test_streaming.py`.

**Consequence for boundary detectors: they must be online.** "The code changed"
and the PerAct gripper/velocity heuristic both are. **UVD is not** — it walks
backward from the final frame, so it is an offline evaluation tool for E3 and must
never appear in a test-time path.

The in-progress event is deliberately absent from the log. That is the right
division of labour: the log carries the past, and the policy already observes the
present directly.

### The trap that would kill the whole claim

BPE merges frequent adjacent pairs. `swing` follows `swing` constantly — that *is* the task — so
**BPE will merge `swing swing` into one token and destroy the count.**

Two guards, use both:
1. **Forbid self-merges** where both halves are the same token (one condition in the merge loop).
2. Run BPE **within event boundaries only**, using subgoal transitions / UVD / the PerAct heuristic
   as hard barriers so merges never cross an event.

The off-the-shelf tokenizer is count-hostile by default. This is not optional.

### Boundary detectors for the barriers and for validation

- **UVD** (ICRA 2024, `github.com/zcczhang/UVD`) — training-free, one line:
  `uvd.get_uvd_subgoals(video, preprocessor_name="vip")` on an `(L,H,W,3)` array. Runs on
  `agentview_rgb` today.
- **PerAct keyframe heuristic** — ~20 lines over `joint_velocities` + `gripper_states`:
  ```python
  small_delta = np.allclose(obs.joint_velocities, 0, atol=0.1)
  stopped = (stopped_buffer <= 0 and small_delta and not next_is_not_final
             and gripper_state_no_change)
  if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
      episode_keypoints.append(i)
  ```
- **SBD** (ICCV 2025, [arXiv:2503.10684](https://arxiv.org/abs/2503.10684)) — prediction-error
  boundaries, explicitly from event-segmentation theory: `if loss − mean(loss_history) > GAP` with
  **GAP = 18**, running mean reset per boundary, lengths pruned to [15, 200]. The published version
  of the novelty gate.

### Naming

Once, offline, per code. Each token gets the label that is modal across its occurrences —
"press the button." A finite codebook makes this bounded; a continuous embedding cannot be named.

---

## 4e. What the first implementation measured (2026-08-21)

Four corrections came out of building it. Each was found by a measurement, and
each invalidated a design choice taken from the literature without checking it
against this data.

**1. The event lives in the action, not in vision.** Feeding vision into the
encoder pulled codes toward *trajectory phase* rather than *motion type*. Removing
vision from the input entirely and clustering action chunks fixed it. Vision
belongs on the *target* side (outcome: "did the grasp succeed"), never the input.

**2. Plain k-means beats the learned tokenizer.** On 40 SwingXtimes episodes:

| | within-event change | boundary P | label MI | n-gram == N |
|---|---|---|---|---|
| k-means K=32 | **11.6 %** | **0.127** | **59.5 %** | **29/40** |
| neural, 4 epochs | 21.4 % | 0.061 | 37.2 % | 0/20 |

`models/kmeans.py` is therefore a first-class path, not a diagnostic. **If a
learned tokenizer cannot beat Lloyd's algorithm, the learning is not earning its
place.** Keep it in every comparison.

**3. The code stream is not a segmentation.** Boundary recall is 0.87–1.00 but
precision is 0.06–0.13, so it over-segments 4–8×. It is a *high-recall candidate
set*. Within-event stability cannot detect this — a code constant over the whole
episode scores a perfect 0 % change rate while segmenting nothing — which is why
`eval/repeatability.py` reports boundary precision and label MI alongside it.
**Do not call a code change an event boundary.**

**4. BPE must run over run *symbols*, not within runs.** Splitting at code changes
makes every span a constant stretch, so every adjacent pair inside it is a
self-pair, which the no-self-merge guard forbids — 0 merges, and an "event log"
that was just the raw stream at 429 tokens per episode. Correct structure:

```
raw     A A A B B C C C A A     transitions
runs    A B C A                 candidate units, over-segmented
BPE     [AB] C A                the selection stage
```

Guard 1 still protects counts; guard 2 becomes "no merges across episodes", which
is automatic. After the fix: 429 transitions → 48 runs → **16.7 event tokens**, a
26× compression, with the most frequent token's count matching N in 19/40 episodes.

### Where it stands

The pipeline runs end to end and every stage is measured. Count accuracy at 48 %
is well above chance and not yet a result. The over/under split (8 over, 13 under
at `min_frequency=15`) shows a genuine tuning optimum between under-merging and
over-merging — the first thing to sweep on GPU.


---

## 4f. Multimodality, measured (2026-08-21)

The tokenizer had drifted into being **OAT with different hyperparameters** —
FSQ over action chunks, register encoder, reconstruction loss, no vision. That
drift came from validating only on SwingXtimes, whose events *are* motions, so
vision looked redundant. Two tasks and a metric fix settle it.

### The metric was wrong first

RoboMME's subgoals name the object: *"pick up the container that hides the red
cube."* On occlusion tasks that object is **hidden by the container** at pick
time, and the colour varies only across episodes (ButtonUnmask: red 0.50/ep,
green 0.30, blue 0.45, at most one pick each). **No code computed from a single
transition can carry it.** Scoring against the raw label measures an impossible
target.

`subgoals.observable_label()` collapses object identity to `<obj>`. On
ButtonUnmask this moved the multimodal reading from 38.0% to 53.9% and *inverted*
the conclusion about whether vision helps. The binding is not lost — supplying it
is exactly the memory log's job. **Event tokens carry observable events; which
cube went under which container comes from earlier entries in the log.** That
division of labour is worth stating in the paper.

### With the fair metric, multimodal wins on both tasks

MI as a fraction of observable-label entropy, k-means K=32, 40 episodes:

| input | ButtonUnmask | SwingXtimes |
|---|---|---|
| action only (**= OAT's input**) | 48.5 % | 74.3 % |
| vision only | 45.2 % | 47.2 % |
| **action + vision** | **53.9 %** | **81.4 %** |

+5.4 and +7.1 points over action-only. **This is the measured answer to "how is
this different from OAT": OAT is action-only.** Vision adds consistently, on both
a motion-defined task and a world-defined one.

Note vision alone is the *weaker* single modality on both tasks, so the gain is
genuinely from the combination rather than from vision carrying everything.

### Method note

Validate on at least two tasks with different structure from M1 onward. Every
design decision that had to be undone — 512 codes, no vision, code-change
boundaries — traces to a SwingXtimes-only measurement. One task cannot tell a
general mechanism from a coincidence of that task's structure.


---

## 4g. Corrected results, with baselines (2026-08-21)

Every headline number in the two sections above was reported on an uncalibrated
scale. Corrected, with baselines on the same line.

### Counting: works, on held-out episodes

Pattern selected on 50 train episodes using train N only, then that **fixed**
pattern's occurrence count evaluated on 50 held-out episodes:

| pattern length | train | **test** | chance (sd) | pattern |
|---|---|---|---|---|
| 1 | 98 % | **100 %** | 35 % (7) | `(13,)` |
| 2 | 98 % | **100 %** | 35 % (7) | `(13, 12)` |
| 3 | 96 % | **100 %** | 35 % (7) | `(26, 13, 12)` |
| 4 | 90 % | 90 % | 32 % (6) | `(0, 16, 26, 13)` |
| 6 | 74 % | 74 % | 30 % (6) | `(8, 0, 16, 26, 13, 12)` |

The patterns are **nested** — one swing is the run-symbol sequence
`0 16 26 13 12`, and any contiguous piece of it counts the swings. So the
length-1 result is not a degenerate shortcut; it is the most robust member of a
real multi-symbol pattern.

Two things to note. This is **action-only k-means with no BPE at all**, so whether
the BPE stage contributes to counting is still an open question. And selecting the
pattern uses train N, which is legitimate supervision for *which* event to count,
but means the system must be told which event matters.

### Event identity: report accuracy, not MI

Predicting the observable event label from the code, cluster→majority-label fitted
on train frames, evaluated on held-out frames:

| input | ButtonUnmask (3 labels) | SwingXtimes (5 labels) |
|---|---|---|
| majority baseline | 49.6 % | 30.0 % |
| action | 77.4 % | 80.9 % |
| vision | 76.6 % | 60.3 % |
| **action + vision** | **80.5 %** | **87.2 %** |

Multimodal gain: **+3.1** and **+6.3** points. Real, modest, and on a readable
scale — unlike MI/H, where the same ButtonUnmask result reads as "53.9 %" and
invites the conclusion that it is half broken. It is not: 80.5 % against a 49.6 %
baseline.

### Four metric failures, one cause

1. `max(token_count) == N` — 48 %, chance 31–37 %. Nothing.
2. `best_repeating_ngram(codes, target=N)` — selected the gram *by* the answer.
   Its "29/40" was meaningless. Now flagged `_CIRCULAR` in the code.
3. Counting reported with no chance baseline for several turns.
4. MI/H read as though it were an accuracy.

All four are the same mistake: a number quoted without its baseline.
`eval.repeatability.label_accuracy()` now returns its baseline alongside the
score so this is harder to repeat.


---

## 5. Experiments

**E1 — does a memory string change behaviour at all? (week 1, before building)**
Hand-write memory sentences ("you have already scooped twice") and prepend them to a VLA on a
counting task. If a *correct hand-written* memory changes nothing, the delivery mechanism is dead and
everything downstream is moot. An afternoon.

**E2 — code consistency.** `P(same code | same subgoal)` vs `P(same code | different subgoal)`. Plus
a repeat test: edit distance between code sequences for N repetitions of the same demonstration.

**E3 — segmentation quality.** BPE token boundaries vs subgoal transitions (F1 with tolerance); UVD
and the PerAct heuristic as alternates.

**E4 — counting.** Probe the count from the token sequence. Baselines: memoryless, a
similarity-merged token bank, a VLM captioner (≈ MemER), a GRU.

**E5 — count extrapolation. THE central experiment (see §7).** Counts beyond training range and
beyond WeaveLA's reported N=3. A pooled latent must encode "how many" in its geometry; an
append-only token sequence just gets longer. **Match at N=3, win at N≥5.**

**E6 — zero-adaptation transfer.** Same memory string into ≥3 VLAs, inference only. Then few-shot.
Expect few-shot > zero-shot; the gap is a result.

---

## 6. Build order

**Week 1.** E1 first. Then: train the tokenizer on `SwingXtimes` alone using precomputed SigLIP
features + actions, dump the raw code stream, and **plot it against the subgoal timeline. Does the
same code recur once per swing?** If yes, everything downstream is bookkeeping. If not, adjust `k`
and vocabulary size before adding machinery.

**Week 2.** Wire `obs_emb` through QueST's `compute_autoencoder_loss`. E2 code consistency. BPE with
both self-merge guards. E3 boundaries.

**Week 3.** E4 counting, then E5 extrapolation across N.

**Week 4.** E6 transfer, then write.

Everything through week 3 is offline — cached features, forward passes, linear probes. **No
simulator, no rollouts, no policy training.** That is what makes four weeks feasible.

---

## 7. Prior art and threats

### WeaveLA — the serious one

**"WeaveLA: Event Driven Cross-Subtask Latent Memory Weaving for Repetitive Robot Manipulation"**,
Zhu et al., 16 Jun 2026, [arXiv:2606.17463](https://arxiv.org/abs/2606.17463). **RoboMME. π₀.₅
backbone. Repetitive manipulation.** Subtask completion is the event trigger; the finished segment is
pooled into latent tokens by query-driven attention and injected into the next subtask. Frozen base
policy. **`SwingXtimes` (N=3): 0 % → 47.8 %.**

Treat "event-triggered memory for repetitive manipulation on RoboMME" as **taken**. What survives:
**discrete vs. continuous**, **append-only log vs. one-step handoff**, **cross-architecture text
transfer**. The wedge is E5 — a latent handoff should not scale in N.

**Read WeaveLA in full before building.** Specifically: does it handle N>3, and does anything in it
actually count? If it counts, the wedge closes and the angle needs to change.

Silver lining: their memory-free baseline is **0 %** on `SwingXtimes`. Counting tasks really are
broken — but the comparison is now against WeaveLA, not a memoryless policy.

### Also close

- **KEMO** ([2606.23589](https://arxiv.org/abs/2606.23589)) — kinematics + visual filtering for event
  detection, keyframes as "temporally ordered memory tokens", cross-attention + gated residual.
  +23.6 % TSR / +34.1 % SCR, real dual-arm, 830–2846-step episodes. But it *selects* task-relevant
  keyframes, and filtering is where counts die. Tokens are keyframe embeddings, not a vocabulary.
- **EventVLA** ([2606.20092](https://arxiv.org/abs/2606.20092)) — keyframe-probability head, commits
  **raw images** to a FIFO buffer (N_max=5), memory re-enters as context. **+40 % over SOTA memory
  VLAs.** No code. *They buffer raw images; we buffer discrete tokens.* State this in the intro.
- **OmniAct** ([2606.27251](https://arxiv.org/abs/2606.27251)) — "event-boundary-driven compression,
  sub-linear context growth", omnimodal agent scope. Mechanism not in the abstract; **pull the PDF
  before writing related work.**
- **VQ-Memory** ([2603.09513](https://arxiv.org/abs/2603.09513)) — VQ-VAE over past *proprioceptive*
  states into discrete latent tokens. Closest on the discretization axis; no vision, no counting.
- **MME-VLA's VLM subgoal predictor** — trained on these same annotations. Distinction to state:
  *subgoals are prescriptive (what to do next); event tokens are a record (what happened, in what
  order, how many times).*

### Confirmed empty

No published work does variable-length discrete event tokens used as policy memory. The
segmentation literature has variable length without codes; the skill-tokenizer literature has codes
without variable length; the memory literature has events without discretization.

---

## 8. Risks

| risk | severity | mitigation |
|---|---|---|
| VLAs ignore the memory string | **fatal** | E1 in week 1; phrase as natural language |
| codes don't recur for repeated events | **fatal** | week-1 plot; adjust `k` and vocab first |
| BPE merges repeats and destroys counts | **fatal** | both guards in §4, non-negotiable |
| WeaveLA already counts | high | read it first; if so, change angle |
| codes capture camera/pose nuisance | high | action stream + E2 diagnostics |
| absolute-vs-relative state fragments codes | medium | transform `ee_states` to start-relative or object-centric frames |
| "this is just prompting" | certain | the contribution is the tokenizer; text is delivery |

## 9. Open questions

- Tokenizer trained per-task or shared across the 16 tasks? Sharing is what the transfer story needs;
  OAT trains per-dataset and does not test sharing.
- One joint vocabulary for vision+action, or two parallel ones? Joint is simpler; separate lets us ask
  which modality carries event identity.
- Does the coarse/fine split from nested dropout actually align with event-type vs. instance
  specifics, or is that wishful? Per-prefix-depth code agreement in E2 answers it.
