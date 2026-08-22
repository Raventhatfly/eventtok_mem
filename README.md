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

`docs/` holds the research plan, prior-art notes and the parked spiking-memory design notes. It is
gitignored — working notes, not part of the published code. What matters for reading the code is in the
module docstrings, which record the measurements and the designs that failed.

## Status

Codebase under construction.

## Archive

This repo began as `ssn_robotic_memory`, an STDP-trained spiking memory bank. That line is parked, not
deleted — `archive/spikebank/` has the implementation, and the design notes live in the (untracked)
`docs/archive/`. Two findings from it survive into the current design: novelty gating as an event-boundary
signal, and nested-dropout ordering for coarse-to-fine event tokens.
