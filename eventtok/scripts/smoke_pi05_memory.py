"""Push a synthetic event log through pi0.5's memory encoder. CPU, no checkpoint.

    JAX_PLATFORMS=cpu python -m eventtok.scripts.smoke_pi05_memory

Catches the failures that would otherwise surface several minutes into a GPU job: a
dtype the typecheck rejects, a budget that disagrees with the slot count, an encoder
whose output is not the LLM width. It also checks the attention mask, because a memory
block the image and action tokens cannot attend to would train quietly to no effect.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROBOMME = Path(os.environ.get("ROBOMME_REPO", "/n/home04/wfy/repos/robomme_policy_learning"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-symbols", type=int, default=40)
    ap.add_argument("--max-log", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--tag", default=None, help="read dims from a built cache instead")
    args = ap.parse_args()

    sys.path.insert(0, str(ROBOMME))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    from ..pi05.config import history_config
    from ..pi05.features import EventMemoryFeatures

    n_symbols, max_log = args.n_symbols, args.max_log
    if args.tag:
        from ..pi05.joint import TASKS
        from ..pi05.tokens import EventLogCache, cache_path

        task = next(t for t in TASKS if cache_path(t, args.tag).is_file())
        c = EventLogCache(task, args.tag)
        n_symbols, max_log = c.n_symbols, c.max_log
        print(f"dims from {task}/{args.tag}: vocab {n_symbols}, budget {max_log}")

    cfg = history_config(n_symbols, max_log)

    import flax.nnx as nnx
    import jax.numpy as jnp
    from omegaconf import OmegaConf

    from mme_vla_suite.models.integration.history_pi0 import make_attn_mask
    from mme_vla_suite.models.representation.percep_mem import PerceptualMemory

    mem = PerceptualMemory(config=OmegaConf.create(cfg), rngs=nnx.Rngs(0),
                           dtype=jnp.float32)

    feat = EventMemoryFeatures(n_symbols, max_log)
    rng = np.random.default_rng(0)
    rows = []
    for b in range(args.batch):
        length = int(rng.integers(0, max_log + 1))
        toks = np.zeros(max_log, dtype=np.int64)
        toks[:length] = rng.integers(0, n_symbols, length)
        ovf = np.zeros(n_symbols, dtype=np.float32)
        ovf[rng.integers(0, n_symbols)] = b  # one batch element has evictions
        rows.append(feat(toks, length, ovf))
    batch = {k: jnp.asarray(np.stack([r[k] for r in rows])) for k in rows[0]}
    for k, v in batch.items():
        print(f"  {k:<18} {v.shape} {v.dtype}")

    tokens, _, _ = mem(batch["static_image_emb"], batch["static_pos_emb"],
                       batch["static_state_emb"])
    print(f"memory tokens: {tokens.shape} {tokens.dtype}")
    assert tokens.shape == (args.batch, max_log, cfg["memory_token_dim"]), tokens.shape
    assert jnp.isfinite(tokens).all(), "non-finite memory tokens"

    # Slots holding the same token must embed identically -- repeats carry the count.
    same = mem(
        jnp.asarray(feat(np.array([3] * max_log), max_log,
                         np.zeros(n_symbols, np.float32))["static_image_emb"])[None],
        batch["static_pos_emb"][:1], batch["static_state_emb"][:1] * 0,
    )[0][0]
    spread = float(jnp.abs(same - same[0]).max())
    print(f"identical tokens, differing slots: max |delta| = {spread:.4f}")
    assert spread > 1e-4, (
        "slots holding the same token embed identically -- the position is not "
        "reaching the encoder, so the log is a set and counts are invisible"
    )

    # Who reads the memory. Upstream's na_mask deliberately blocks the image tokens
    # from attending to the memory block and lets the action expert through -- the
    # memory informs the policy, not the vision tower. Assert both halves: if the
    # action tokens cannot see it either, training would quietly do nothing.
    n_img, n_act = 8, 4
    total = max_log + n_img + n_act
    input_mask = jnp.ones((1, total), dtype=bool)
    ar = jnp.asarray(
        [False] * max_log                       # memory: attends within the prefix
        + [True] + [False] * (n_img - 1)        # images
        + [True] + [False] * (n_act - 1)        # action expert
    )
    na = jnp.asarray([False] * max_log + [True] * n_img + [False] * n_act)
    attn = make_attn_mask(input_mask, ar, na)
    by_image = bool(attn[0, max_log : max_log + n_img, :max_log].any())
    by_action = bool(attn[0, max_log + n_img :, :max_log].any())
    print(f"memory read by: images={by_image} action={by_action}")
    assert by_action, "the action expert cannot see the memory; it would train to nothing"
    assert not by_image, (
        "images now attend to memory -- upstream's na_mask changed, and the event "
        "memory is no longer wired the way the MovieChat baselines are"
    )

    print("OK")


if __name__ == "__main__":
    main()
