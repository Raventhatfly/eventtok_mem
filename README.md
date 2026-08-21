# eventtok_mem

**Event tokens as robot policy memory.**

Tokenize a robot's own experience into a short sequence of **discrete event tokens**, keep them in an
**append-only** log, and feed that log back to the policy as context. The log is a sentence in a learned
language, so a language model is the natural consumer.

## The claim

> Boundary-coupled variable-length **quantization** feeding an **append-only, count-preserving** event
> token log, delivered as **text** so it transfers across policies without adaptation.

Two supporting arguments:

1. **Token sequences preserve repetition; consolidation destroys it.** Methods that merge redundant
   entries discard exactly what counting needs. Ask a captioner to summarise three scoops and it says
   "scooping"; a token log says `scoop scoop scoop`. *Compression is fine — deduplication is the killer.*
2. **Text-valued tokens transfer with zero adaptation.** Every modern VLA has a language channel, so the
   same memory string drops into π₀.₅, OpenVLA, RDT with no adapter, no projector, and no shared
   embedding space.

## How it works

```
transitions (k=20 frames)     vision features + action chunk
        │
   transformer + register tokens  →  FSQ (512 codes)
        │                            trained by dual heads:
        │                              A: (feat_t, code) → feat_{t+k}   cosine
        │                              B: code           → action chunk  L1
        │                            never pixel reconstruction
   per-transition code stream
        │
   BPE  (self-merges forbidden; event boundaries are hard barriers)
        │
   variable-length event tokens  →  named by modal subgoal  →  prompt text
```

Counting falls out: two occurrences of an event are two occurrences of a token, in sequence. No
amplitude, no decay, no retrieval.

## Docs

| file | contents |
|---|---|
| [`docs/EVENT_TOKENIZER_PLAN.md`](docs/EVENT_TOKENIZER_PLAN.md) | the research plan — design, experiments, prior art, risks |
| [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md) | positioning and the novelty threats (WeaveLA, KEMO, EventVLA) |
| [`docs/_archive_event_tokenizer_v1.md`](docs/_archive_event_tokenizer_v1.md) | exploratory version, kept for the reasoning trail |
| `docs/archive/` | the parked spiking-memory line (see below) |

## Status

Codebase under construction. See the plan for milestones M0–M4.

## Archive

This repo began as `ssn_robotic_memory`, an STDP-trained spiking memory bank. That line is parked, not
deleted — `archive/spikebank/` has the implementation and `docs/archive/` the design notes and measured
results. Two findings from it survive into the current design: novelty gating as an event-boundary
signal, and nested-dropout ordering for coarse-to-fine event tokens.
