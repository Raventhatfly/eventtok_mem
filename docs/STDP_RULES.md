# STDP in practice — what works, what plateaus, and what our rule got wrong

*Condensed from a 2026-08-18 sweep. The first section is a direct post-mortem on `EXPERIMENTS.md` E2.*

## 0. Why E2 failed — three violated preconditions

Our `stdp.py` breaks the conditions under which STDP is *provably* doing something useful.

| what we did | what the literature requires | source |
|---|---|---|
| linear soft bound `(w_max − w)` | **exponential** weight-dependent potentiation `Δw ∝ exp(−w)`, or power-law with **μ ≈ 0.2** | Nessler 2013; Gütig 2003 |
| `a_minus = 0.6` (LTD/LTP ≈ 0.6) | Diehl & Cook's released code uses `nu_pre=1e-4, nu_post=1e-2` → **LTD/LTP ≈ 1:100** | Diehl & Cook 2015 source |
| **hard** k-WTA | **soft, stochastic** WTA with temperature `T` — it is the *normalisation*, not a sparsity gadget | Nessler 2013; SoftHebb 2022 |
| dense continuous input | **population-coded** inputs (the EM theorem assumes them) | Nessler 2013 |

**Nessler, Pfeiffer, Buesing, Maass, "Bayesian Computation Emerges in Generic Cortical
Microcircuits through STDP", *PLoS Comput. Biol.* 2013**
([link](https://doi.org/10.1371/journal.pcbi.1003037)) is the keystone: each output spike from a
soft-WTA circuit **is** an E-step, and the STDP update that follows **is** a stochastic online
M-step. Weights converge to `w* = log P(pre fired within τ before post | post fired)`. The
`exp(−w)` factor is exactly what makes the fixed point a *normalised conditional probability*
rather than a runaway correlation — which is why our version densified to 99.6 %.

**Fix list for E3, in order:** (1) `exp(−w)` or μ=0.2 power-law potentiation; (2) LTD/LTP ≈ 1:100;
(3) softmax WTA with temperature instead of hard top-k; (4) population-code the proprio channel.

## 1. The single most important theoretical result for this project

**Rao & Sejnowski, "Spike-Timing-Dependent Hebbian Plasticity as Temporal Difference Learning",
*Neural Computation* 13(10):2221–2237, 2001**
([PDF](https://faculty.cc.gatech.edu/~isbell/reading/papers/rao_sejnowski_nc_2001.pdf)).

**An asymmetric STDP window is a TD(0) rule on the postsynaptic membrane potential.** Consequence:
**recurrent excitatory STDP learns to predict input sequences**, and after learning the network
fires *in advance* of the stimulus.

→ This is the citation for `DESIGN.md` §5.3's claim that STDP learns **transitions, not features**.
It also means **we do not need a separate prediction loss** — recurrent activity *is* an implicit
next-state prediction, readable as anticipatory firing. Set the LTP window to one control step.
Caveat: the equivalence is derived in a linearised one-step-ahead setting, horizon ≈ 1 step.

## 2. Recurrence is not optional — the numbers

Local learning rules lose badly to BPTT **without** recurrence and are nearly at parity **with** it:

| SHD benchmark | BPTT | e-prop | DECOLLE | gap |
|---|---|---|---|---|
| feedforward | **75.85 %** | 63.04 % | 58.55 % | **−12.8 / −17.3 pts** |
| **recurrent** | **83.23 %** | 80.79 % | — | **−2.4 pts** |

→ The entire viability of the gradient-free story rests on the recurrent connections. This is the
strongest single argument for the SpikeBank architecture over a feedforward spiking encoder.

## 3. What STDP actually is, as an algorithm

| knob setting | resulting algorithm |
|---|---|
| hard WTA + additive STDP | online **k-means / vector quantisation** |
| soft *stochastic* WTA + `exp(−w)` potentiation | **stochastic online EM** for a mixture model (Nessler 2013) |
| + plastic lateral inhibition + threshold homeostasis w/ spike-count target | **sparse coding** (SAILnet, LCA) |
| + intrinsic plasticity toward an exponential rate target | **ICA** (Savin, Joshi, Triesch 2010) |

**Pick a point deliberately.** For a memory bank with retrievable discrete episodes → the k-means/EM
end (prototype neurons). For a compressed recurrent state code → the sparse-coding end
(soft competition, p ≈ 3–10 co-active units per frame). *These are different systems; one layer
cannot be both.* Our design wants the sparse-coding end for the readout and the EM end for the
associative store — which argues for **two differently-tuned populations**, not one.

Key refs: **SoftHebb** (Moraitis et al., *Neuromorph. Comput. Eng.* 2022,
[doi](https://doi.org/10.1088/2634-4386/aca710)) — soft WTA beats hard WTA on both accuracy
(MNIST 96.31 % vs 95.78 %) **and** noise robustness (65 % vs ~40 % at ε=64/255, backprop ~0 %);
temperature `T` interpolates the whole spectrum. **SAILnet** (Zylberberg et al. 2011) — three local
rules only: Oja feedforward + *learned* lateral inhibition + threshold homeostasis with an explicit
**spike-count target p** (interpretable sparsity knob).
**Habenschuss et al. NIPS 2012** — homeostasis is *EM with posterior constraints*, i.e.
theoretically mandatory, not a hack.

## 4. Honest ceilings

- **Single-layer STDP vs an autoencoder of equal width**: CIFAR-10 **−10.3 pts**, CIFAR-100
  **−12.5 pts**, STL-10 −3.1 pts (Falez et al., *Pattern Recognition* 2019,
  [arXiv:1901.04392](https://arxiv.org/abs/1901.04392)). Their diagnosis: **inhibition fails to
  enforce diversity** — max pairwise feature coherence **0.999**. WTA makes the code sparse without
  making it diverse. → *Measure pairwise cosine coherence between bank weight vectors during
  training; if it approaches 1.0 the inhibition is decorative.*
- **Best feedback-free local learning ever** (SoftHebb, 5 layers, ICLR 2023,
  [arXiv:2209.11883](https://arxiv.org/abs/2209.11883)): CIFAR-10 80.3 %, **ImageNet top-1 27.3 %**.
  That is the ceiling. → *Keep the vision encoder frozen and pretrained; let STDP work only on the
  temporal/memory problem, where it is competitive.*
- **Pure STDP vs R-STDP on NORB: 66 % → 88.4 % (+22.4 pts)** (Mozafari et al., *IEEE TNNLS* 2018,
  [arXiv:1705.09132](https://arxiv.org/abs/1705.09132)). Mechanism: unsupervised STDP learns what
  **repeats most**; the third factor learns what is **diagnostic**. → *On a robot, "frequent" means
  floor texture and idle proprioception. Budget the third factor from day one, not as phase 3.*
- **Uniform/unstructured feedback is catastrophic**: e-prop TIMIT transcription 24.7 % → **60 % PER**
  with uniform feedback; SuperSpike with random feedback is "worse than a network without hidden
  units" on hard temporal tasks. → *Keep the readout feedback structured.*
- **Nobody has published an STDP-trained spiking world model.** The working spiking forward models
  (Huebotter et al. 2025, [arXiv:2509.05356](https://arxiv.org/abs/2509.05356), Franka Panda) use
  **surrogate-gradient BPTT**. Their own words: predictive-model SNNs are "in their infancy".
- **Zero Transformer comparisons exist** anywhere in the biologically-plausible sequence-learning
  literature. Ours would be among the first — which is both the opportunity and the exposure.

## 5. Sequence learning — the three blueprints worth copying

1. **HTM temporal memory** — Hawkins & Ahmad, *Front. Neural Circuits* 2016; Cui, Ahmad, Hawkins,
   *Neural Computation* 2016 ([arXiv:1512.05463](https://arxiv.org/abs/1512.05463)).
   **2048 mini-columns, top 2 % active, single-pass always-on unsupervised**, multiple cells per
   column give *context* so the same input is represented differently depending on history —
   variable-order sequence memory. On NYC taxi it **matches a hand-tuned LSTM** while using no
   tuning and one pass, recovers far faster after distribution shift, and is unaffected by **30 %
   cell death**. It loses on grammar tasks (98.4 % vs LSTM 100 %).
   **Their stated limitation is our project:** *"we have only tested HTM on low-dimensional
   categorical or scalar data streams… may require dimensionality reduction and feature extraction
   first."* Our frozen-encoder + sparse-code front end **is exactly that missing piece** — which
   also means the burden falls entirely on the encoder to produce **repeatable** sparse codes.
2. **Spiking HTM** — Bouhadjar, Wouters, Diesmann, Tetzlaff, *PLoS Comput. Biol.* 2022
   ([arXiv:2111.03456](https://arxiv.org/abs/2111.03456)). HTM made actually spiking: dendritic
   action potentials as the predictive signal, spike-timing-dependent **structural** plasticity,
   homeostatic control of synapse growth. Prediction error → 0 after ~30 episodes.
   **Hard constraint: usable inter-stimulus interval 10–75 ms (≈15–100 Hz)** — compatible with a
   10 Hz control loop only at the slow end. Learns **order, not duration**.
3. **Clock + readout** — Maes, Barahona, Clopath, *PLoS Comput. Biol.* 2020
   ([arXiv:1907.08801](https://arxiv.org/abs/1907.08801)). Separate the **timing backbone**
   (clustered recurrent net that learns *when*) from the **content readout** (learns *what*).
   2400 E / 600 I in 30 clusters × 80, ~15 ms/cluster. Temporal variability grows as **√t**.
   → *Feed observations into the readout, not the clock.* This directly addresses blueprint 2's
   "no duration" limitation.

Also: **Izhikevich 2006 polychronization** — give the recurrence a **distribution of axonal
delays**, not one Δt; delays encode "what follows what" in relative timing. And
**Lu & Wu 2024** ([arXiv:2404.02729](https://arxiv.org/abs/2404.02729)) **proves hidden neurons are
necessary** to store arbitrary sequences → over-provision the bank beyond observation dimensionality.

## 6. Encoding and hyperparameters

**Encoding comparison under STDP-only training** (Guo et al., *Front. Neurosci.* 2021,
[doi](https://doi.org/10.3389/fnins.2021.638474)): TTFS gives the best accuracy/latency/energy
(88.57 %, 20 ms inference, 1.5×10⁸ SOPs vs rate's 150 ms / 9.9×10⁸); **phase coding is most robust
to input noise**; **burst is most robust to hardware faults and converges fastest**; **rate is worst
under synaptic faults**.

→ **Recommendation:** vision embeddings → **latency/rank-order over a per-frame normalised
embedding** (rank-order is *scale-invariant*, so encoder-output drift across scenes is free
robustness); **whiten** rather than DoG-filter (Falez et al. IJCNN 2020,
[arXiv:2002.10177](https://arxiv.org/abs/2002.10177)). Proprioception → **Gaussian
population/receptive-field coding**, which is exactly the format Nessler's EM theory assumes.

**Verified Diehl & Cook defaults** (from released source, cross-checked against BindsNET):
`tc_pre_ee=20 ms`, `tc_post_1=20 ms`, `tc_post_2=40 ms` (triplet), `nu_pre=1e-4`, `nu_post=1e-2`
(**1:100**), `wmax=1.0`, `exp_ee=0.2` (power-law soft bound), `theta_plus=0.05 mV`,
`tc_theta=1e7 ms`, per-neuron **L1 input-weight sum held at 78**, τ_m 100 ms (excitatory),
refractory 5 ms.

**Three deliberate deviations for a robot:**
1. **Shrink the time constants.** The 100 ms τ_m and 350+150 ms presentation window assume a static
   frame. At 10–30 Hz use τ_m ≈ 20–30 ms and a 30–100 ms window, or consecutive observations blur.
2. **Shorten `tc_theta` drastically**, 10⁴ s → **10²–10³ s**. 10⁴ s is "never forgets", tuned for
   i.i.d. MNIST; a robot's distribution is non-stationary and the bank must reallocate neurons.
3. **Keep** the per-neuron L1 norm and the μ=0.2 power-law bound. Those are what stop it exploding.

**The dead-neuron problem is worse for a robot than for MNIST** because the stream is temporally
correlated — the robot stares at one scene for seconds, so one neuron wins thousands of consecutive
updates. Mitigations: shuffle/replay from a short buffer before STDP sees it, and shorten `tc_theta`
so the threshold can respond within a single episode.

**Five-layer stability defence, with separated timescales** (the separation *is* the engineering):
instantaneous WTA → fast **plastic inhibition (must be faster than feedforward STDP)** → medium
power-law soft bounds → slow L1 renormalisation → slowest adaptive threshold → a rate-floor rescue
for silent neurons. Caveat: essentially all of this theory is **feedforward-WTA**; recurrent STDP
stability is an extrapolation we must validate empirically (Zenke, Agnes, Gerstner, *Nat. Commun.*
2015 is the closest recurrent result, and it says: fast heterosynaptic term or you blow up).

## 7. Net read

STDP-trained spiking sequence memory is **well-supported** for robustness, single-pass online
adaptation, fault tolerance, and recovery from non-stationarity — precisely the properties a
streaming robot needs — and **poorly supported** for raw predictive accuracy on high-dimensional
input, where it has essentially never been benchmarked.

That maps cleanly onto `PRIOR_ART.md` §8: claim **online gradient-free write and adaptation**, not
raw success rate. Design the **encoder** as the make-or-break component, keep STDP on the temporal
problem, add a prediction-error third factor early, and treat recurrence as non-negotiable.
