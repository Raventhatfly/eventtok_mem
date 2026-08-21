# Memory mechanisms in SNNs — what the literature says we must build

*Condensed from a 2026-08-18 sweep. Ordered by how much each changes our design.*

## The ten that change the build

1. **Set the decay constants from the LMU/HiPPO basis, not by hand.**
   *Legendre Memory Units*, Voelker, Kajić, Eliasmith, NeurIPS 2019
   ([proceedings](https://papers.nips.cc/paper/9689-legendre-memory-units-continuous-time-representation-in-recurrent-neural-networks)).
   Derives the **provably optimal** linear system for representing a sliding window of length θ:
   d coupled ODEs whose state is the Legendre expansion of the recent past. 10⁵-step
   dependencies; ~2 orders of magnitude more memory capacity than LSTM; **has a published
   spiking / Loihi implementation** ([Nengo](https://www.nengo.ai/nengo-examples/loihi/lmu.html)).
   → Initialise `β` from the LMU `A` matrix's eigenvalues for a target window θ, instead of the
   hand-picked log-grid in DESIGN.md §5.1. Keep the log-grid as the ablation.

2. **Wire the banks as a non-normal feedforward chain (fast→medium→slow), not block-diagonal.**
   *Memory traces in dynamical systems*, Ganguli, Huh, Sompolinsky, PNAS 2008
   ([link](https://www.pnas.org/doi/10.1073/pnas.0804451105)). Fisher Memory Curve analysis:
   total memory is **bounded ≈ 1 for normal (symmetric-ish) networks**; only **non-normal**
   (hidden-feedforward / chain) connectivity achieves memory **extensive in N**.
   → **Direct correction to DESIGN.md §5.1**, which used block-diagonal recurrence for
   interpretability. Interpretability is not worth an O(1) memory ceiling. Add
   fast→slow feedforward projections between sub-banks; keep recurrence within-bank.

3. **Two coupled decays per slow neuron beat one long τ.**
   *DEXAT — adaptive threshold neuron with nanodevice implementation*, Shaban, Bezugam, Suri,
   *Nature Communications* 12:4234 (2021) ([link](https://www.nature.com/articles/s41467-021-24427-8)).
   Double-exponential adaptive threshold; **1200 ms of store-recall memory from constants of
   30 ms and 300 ms** — far better numerically conditioned than one 1 s τ.
   → Our τ=60 s bank has `β=0.998`, which is exactly the badly-conditioned regime warned about
   (and in low precision `exp(-Δt/τ)` rounds to 1.0, silently making it a perfect integrator).
   Replace with two coupled shorter decays.

4. **Adaptive threshold / SFA gives seconds of memory at zero synaptic cost.**
   *LSNN*, Bellec, Salaj, Subramoney, Legenstein, Maass, NeurIPS 2018
   ([arXiv:1803.09574](https://arxiv.org/pdf/1803.09574)); *Spike frequency adaptation supports
   network computations on temporally dispersed information*, Salaj et al., **eLife 2021**
   ([link](https://elifesciences.org/articles/65459)) — SFA **alone, with no plasticity at test
   time**, solves 12AX and relational-reasoning tasks. Biological SFA has power-law components
   lasting **>20 s** (Pozzorini et al., *Nat. Neurosci.* 2013).
   → Already in our design (`a`, `θ`); the literature says *lean on it harder* than on long τ,
   because a very slow membrane produces dead neurons whereas SFA does not.

5. **Standard homeostasis is too slow to stabilise STDP.**
   *The temporal paradox of Hebbian learning and homeostatic plasticity*, Zenke, Gerstner,
   Ganguli, *Curr. Opin. Neurobiol.* 2017 ([PDF](https://ganguli-gang.stanford.edu/pdf/16.HebbianLearningHomeostaticPlasticity.pdf)).
   Homeostasis acts over hours–days; Hebbian instability over seconds. **This is the single most
   likely thing to break a het-LIF + STDP bank.** Need a *rapid* compensatory process: fast
   inhibitory plasticity, per-neuron threshold homeostasis, or per-batch weight normalisation.
   → Validated empirically here: our first run had homeostasis at rate 0.01 and the bank was
   silent for 100+ steps. Raising it to 0.5 fixed it (see `EXPERIMENTS.md`).

6. **Plain additive Hebbian writes overflow; use the delta rule.**
   *Linear Transformers Are Secretly Fast Weight Programmers*, Schlag, Irie, Schmidhuber,
   ICML 2021 ([PMLR](https://proceedings.mlr.press/v139/schlag21a.html)); *Using Fast Weights to
   Attend to the Recent Past*, Ba, Hinton, Mnih, Leibo, Ionescu, NeurIPS 2016
   ([arXiv:1610.06258](https://arxiv.org/abs/1610.06258)).
   An outer-product Hebbian store saturates once stored keys exceed the key dimension. The fix is
   to write `value − currently_retrieved_value`.
   → Note the equivalence worth putting in the paper: **a decaying outer-product STDP store read
   out multiplicatively ≈ fast weights ≈ linear attention.** One λ per sub-bank gives
   multi-horizon attention for free. That is a much stronger framing than "STDP learns features".

7. **Bounded-synapse STDP is a palimpsest with ~log N capacity.**
   Amit & Fusi, *Neural Computation* 1994 ([link](https://direct.mit.edu/neco/article-abstract/6/5/957/5822/));
   Fusi & Abbott, *Nat. Neurosci.* 2007 ([link](https://www.nature.com/articles/nn1859)) — hard
   bounds give lifetime ∝ (states)² but only under knife-edge LTP/LTD balance; **soft bounds are
   robust but only linear**. Benna & Fusi, *Nat. Neurosci.* 2016
   ([link](https://www.nature.com/articles/nn.4401)) — a chain of coupled variables per synapse
   gives **~N/log N capacity and power-law forgetting**.
   → Use **soft/multiplicative bounds** (already in `stdp.py`). If the robot runs long enough to
   need graceful forgetting, add a Benna–Fusi ladder on the consolidated weight.

8. **Two-stage write: STDP sets a tag, a global signal consolidates it.**
   *Making memories last: synaptic tagging and capture*, Redondo & Morris, *Nat. Rev. Neurosci.*
   2011 ([link](https://www.nature.com/articles/nrn2963)); *Solving the distal reward problem
   through linkage of STDP and dopamine signaling*, Izhikevich, *Cerebral Cortex* 2007
   ([link](https://academic.oup.com/cercor/article/17/10/2443/314939)) — eligibility trace
   τ ≈ 1–2 s × a later global signal.
   → Without this, every idle-motion correlation gets burned into the bank. This is the
   principled version of DESIGN.md §5.4's third factor, and `e-prop`
   (Bellec et al., *Nat. Commun.* 2020, [link](https://www.nature.com/articles/s41467-020-17236-y))
   is its gradient-approximating cousin: eligibility × a top-down learning signal, forward in time.

9. **Clear the activity-silent state between episodes.**
   Barbosa et al., *Nat. Neurosci.* 2020 ([link](https://www.nature.com/articles/s41593-020-0644-4)):
   an uncleaned silent trace produces measurable **serial bias**.
   → In robot terms: the previous object's pose leaks into the current grasp. Explicit reset
   between episodes, and an ablation that measures the leak.

10. **Don't ask one bank for both memory and nonlinearity.**
    *Information processing capacity of dynamical systems*, Dambre, Verstraeten, Schrauwen,
    Massar, *Sci. Rep.* 2012 ([link](https://www.nature.com/articles/srep00514)): total capacity
    = number of independent state variables, and **memory depth trades off against nonlinearity
    within that fixed budget**.
    → Keep the slow banks near-linear (low firing threshold margin, weak recurrence) and let the
    fast banks do feature extraction. Argues against a homogeneous soup.

## Hard ceilings to design against

| Mechanism | Timescale | Capacity | Dominant failure |
|---|---|---|---|
| LIF membrane decay | 1–3 τ_m | ≤ N samples (Jaeger MC bound) | vanishing trace; `exp(-Δt/τ)→1` numerics |
| Heterogeneous τ | spans decades | ↑ vs. homogeneous (Chakraborty & Mukhopadhyay, **ICLR 2023**, [arXiv:2302.11618](https://arxiv.org/abs/2302.11618)) | wrong τ **span**, not wrong values |
| Adaptive threshold / SFA | 0.2–3 s (>20 s power-law in biology) | ~1 scalar/neuron | abrupt failure past ~3τ_a; threshold runaway |
| Short-term plasticity | τ_f≈1 s, τ_d≈0.2–0.8 s | **≈ τ_d/τ_current ≈ 4 items** (Mi, Katkov, Tsodyks, *Neuron* 2017) | serial bias; distractor capture; volatile |
| Discrete attractor | indefinite | 0.138N dense; C/[a·ln(1/a)] sparse (CA3); ∝N low-rank spiking | spurious mixtures; **catastrophic blackout** past capacity |
| Ring/bump attractor | indefinite | 1 continuous variable | diffusive **drift** — unbounded pose error over minutes |
| Reservoir / LSM | ~10²–10³ ms fading | MC ≤ N; IPC = N split memory↔nonlinearity | edge-of-chaos brittleness; input-distribution shift |
| LMU / SSM basis | window θ, 10⁵ steps shown | d ODEs ⇒ order-d window reconstruction | needs linear dynamics; spiking nonlinearity degrades the basis |
| Fast weights / linear attn | λ-decay, 10–10² steps | ≈ key dimension | key collision → delta rule |
| Bounded-synapse STDP | ongoing overwrite | **~log N** dense; √N balanced hard bounds | overwriting; LTP/LTD imbalance |
| Benna–Fusi ladder | seconds → lifetime, power-law | √N → ~N/log N | consolidates noise without gating |
| Learned axonal delays | exact shift | lossless within kernel | fixed horizon (hard wall) |

**Note the MC ≤ N ceiling** (Jaeger 2001): a bank of N neurons cannot linearly retain more than
N past samples, however τ is chosen. Size the bank to `horizon / dt` — for a 60 s horizon at
10 Hz that is 600, so our N_total = 1280 is the right order, not generous.

## Two implementations to read before writing more code

- **Spiking CA3 content-addressable memory with learn/forget/recall**, Casanueva-Morato et al.,
  *Neural Networks* 2024 ([arXiv:2310.05868](https://arxiv.org/html/2310.05868),
  [code](https://github.com/dancasmor/An-aproach-to-a-spike-based-Content-Addressable-Memory-bio-inspired-in-the-Hippocampus)),
  validated on SpiNNaker. Also their robot-navigation version
  ([arXiv:2305.12892](https://arxiv.org/abs/2305.12892)) — a spiking memory in a real-time
  closed control loop.
- **HRSNN — heterogeneous LIF + heterogeneous STDP on temporal video**, Chakraborty &
  Mukhopadhyay, *Front. Neurosci.* 2023
  ([link](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.994517/full)).
  Our stack minus the diffusion readout — take the hyperparameter ranges from here.

Their companion ICLR 2023 paper gives the objective for choosing the τ split:
**maximise memory-capacity ÷ spike-rate**, and make the *STDP window* heterogeneous too, not just
the membrane (already in `stdp.py`: `alpha` scales with each bank's τ).

## Training-loop framing

Complementary Learning Systems (McClelland, McNaughton, O'Reilly, *Psych. Review* 1995) maps
cleanly onto this project: **the spiking bank is the fast sparse episodic store, the Diffusion
Policy is the slow cortex, and replay from bank → policy is the consolidation channel.** A
spiking implementation of exactly this loop exists (*Semantization of memories in a
hippocampal–cortical SNN*, *Neurocomputing* 2025,
[link](https://www.sciencedirect.com/science/article/pii/S0925231225009956)).
