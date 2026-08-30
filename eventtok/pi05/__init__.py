"""Event-token memory for pi0.5 on RoboMME.

pi0.5 is **reactive** -- the RoboMME paper describes it as operating "only on the
current observation" -- so it cannot know how many times it has already done
something. The paper names the resulting failure directly: *state aliasing*, where a
policy repeats a sub-task because the visual state reset. That is the gap this package
fills, and it is a genuine information gap rather than a tidier presentation of
something the policy already has.

Scope discipline, from the benchmark's own numbers: reactive pi0.5 gets 21.5 TSR, and
two of the four memory-augmented baselines score *worse* (HiF-VLA 16.9, MemoryVLA
15.0). Attaching memory is not a free win. MemER is 27.3 and PrediMem 38.5, so those
are the comparisons that matter, not pi0.5's 21.5.

**Nothing here modifies robomme_policy_learning.** The model is subclassed and the
input transform wrapped, so the upstream repo stays untouched.

How the memory enters, and why. pi0.5 already writes its discretised state into the
prompt as text (`"Task: ..., State: 12 87 3 ...;"`), which makes text injection the
house pattern -- but event token ids carry no meaning in Gemma's embedding table, so
`"18 22 18"` would make the model learn a mapping to vectors that encode *the number
eighteen*. Images avoid this by bypassing the vocabulary entirely: SigLIP patches are
projected to 2048-d and inserted as embeddings. Event tokens follow the image path for
the same reason -- a learned embedding table, appended to the prefix.
"""
