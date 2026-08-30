"""Emit the history YAML whose dimensions match a built event cache.

    python -m eventtok.scripts.write_pi05_config --tag joint

Generated rather than hand-written because ``memory_feature.img.input_dim`` must equal
the vocabulary size exactly: the one-hot is the embedding lookup, so a mismatch is a
silent shape error at best and a wrong token at worst.
"""

from __future__ import annotations

import argparse

from ..pi05 import config as cfg
from ..pi05.joint import TASKS
from ..pi05.tokens import EventLogCache, cache_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="joint")
    ap.add_argument("--integration-type", default="context",
                    choices=["context", "modulation", "expert"])
    ap.add_argument("--pos-dim", type=int, default=64)
    args = ap.parse_args()

    built = [t for t in TASKS if cache_path(t, args.tag).is_file()]
    if not built:
        raise SystemExit(f"no caches with tag {args.tag!r}")
    sizes = {}
    for t in built:
        c = EventLogCache(t, args.tag)
        sizes.setdefault((c.n_symbols, c.max_log), []).append(t)
    if len(sizes) > 1:
        raise SystemExit(
            "caches disagree on vocabulary: "
            + "; ".join(f"{k} -> {len(v)} tasks" for k, v in sizes.items())
            + "\nA single policy over several tasks needs one vocabulary; build with "
            "eventtok.scripts.build_pi05_joint."
        )

    out = cfg.write_for_cache(
        built[0], args.tag,
        integration_type=args.integration_type, pos_dim=args.pos_dim,
    )
    (n_symbols, max_log), = sizes
    print(f"{len(built)}/16 tasks, vocab {n_symbols}, budget {max_log} -> {out}")
    print(f"pass it as --model.history_config={out}")


if __name__ == "__main__":
    main()
