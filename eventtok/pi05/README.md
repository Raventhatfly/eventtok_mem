# pi0.5 + RoboMME

The event log as pi0.5's memory, with nothing in `robomme_policy_learning` edited.

## Why the memory slot, not the text prompt

pi0.5 writes its discretised robot state into the text prompt, so spelling the log into
the prompt would be the house pattern. It is still wrong here. A token id rendered as
`"18"` is looked up in Gemma's embedding table and lands on the vector for *the number
eighteen* — a quantity, not a motion. Images avoid this by bypassing the vocabulary
entirely: SigLIP patches are projected straight to 2048-d and spliced into the prefix.
Event tokens take the image path for the same reason.

## Where it attaches

```
RoboMMEDataset.__getitem__          <- replaced: static_* from the event cache
  -> RepackTransform                   unchanged, already forwards static_*
  -> RoboMMEInputs, DeltaActions       unchanged
  -> HistAugObservation.from_dict      unchanged, already carries static_*
  -> HistoryPi0.embed_prefix           unchanged
       embed_memory -> PerceptualMemory -> prepended before the image tokens
```

`PerceptualMemory` is generic: it concatenates `[img, silu(pos_proj(pos)),
silu(state_proj(state))]` and applies one linear map to 2048. No MovieChat logic lives
there — the merging happens on the data side, upstream of this point. So the event log
rides the existing wiring, and MovieChat, token-dropping and event memory end up
differing **only in what fills the slots**. That is the comparison worth reporting.

## The encoding

| field | contents | why |
|---|---|---|
| `static_image_emb` | one-hot over the event vocabulary | one-hot × linear *is* an embedding table, so `encoder_static`'s rows are the learned event embeddings |
| `static_pos_emb` | sinusoid of the slot index | the prefix is bidirectional; without it the log is a set and three swings read as one |
| `static_state_emb` | `log1p` of the eviction tally | the window is bounded, this keeps the count exact past it |
| `static_mask` | which slots are real | |

`budget` is 64 against token-dropping's 512, and the longest observed log across all 16
tasks is 48 tokens, so no demonstration frame overflows. Overflow is reached in failing
rollouts, which is why the tally exists.

## Two seams, no fork

* `get_history_config` builds its path with `os.path.join(<their dir>, arg)`, which
  returns `arg` unchanged when it is absolute — so a YAML in this repo loads as-is.
* `dataloader.py` binds `RoboMMEDataset` with `from ... import`, so rebinding that one
  name in that module is the whole data-side change.

## Order of operations

```bash
# 1. one vocabulary for all 16 tasks -- per-task vocabularies (19-46 symbols) make
#    token 7 mean a different motion in every task
sbatch slurm/pi05_joint.sbatch

# 2. the history YAML, generated so its dims cannot drift from the cache
python -m eventtok.scripts.write_pi05_config --tag joint

# 3. train, and the control alongside it
EVENT_MODE=log   sbatch slurm/pi05_event_train.sbatch
EVENT_MODE=wrong sbatch slurm/pi05_event_train.sbatch
```

`wrong` feeds another episode's log. If it trains to the same success rate, the policy
is not reading the memory and the `log` number means nothing.

## What is not yet known

* Whether one 16-task vocabulary holds up. Per-task fits gave 19-46 symbols; if the
  joint fit needs far more, the per-task label-accuracy results were measuring per-task
  overfitting.
* 16-68% of frames have an empty log — the first chunk cannot close before frame 20 and
  BPE needs a few runs after that. Event memory can only help after that point.
* Whether the log helps at all. Every result so far is offline; success rate is not.
