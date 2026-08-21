"""The M1 go/no-go figure: code stream against the annotated subgoal timeline.

    python -m eventtok.scripts.plot_timeline --ckpt ... --episode 1000

The question this answers, and it is the only question that matters at this
stage: **does the same code recur once per swing?** If the code stream lines up
with the annotated events, everything downstream is bookkeeping. If it does not,
no amount of BPE or prompt engineering rescues it, and the fix is `k` and the
vocabulary size rather than more machinery.

Three rows:
  1. raw per-transition code stream
  2. the streaming event log (spans after boundary detection)
  3. ground-truth subgoal segments, with ordinals stripped so repetitions share a
     colour -- otherwise "for the second time" reads as a different event and a
     correctly working tokenizer looks broken
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .. import paths
from ..bpe.streaming import CodeChangeBoundary, StreamingTokenizer, identity_span_encoder
from ..data import subgoals as sg
from ..data.index import RoboMMEIndex
from ..data.meta import TaskMeta
from ..data.robomme import TransitionDataset
from .tokenize_episodes import load_model, stream_for_episode


def _colour_map(keys) -> dict:
    cmap = plt.get_cmap("tab20")
    return {k: cmap(i % 20) for i, k in enumerate(dict.fromkeys(keys))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="SwingXtimes")
    ap.add_argument("--episode", type=int, default=1000)
    ap.add_argument("--min-span", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    paths.check_root()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = load_model(args.ckpt, device)

    index = RoboMMEIndex()
    ep = index[args.episode]
    meta = TaskMeta(args.task)
    dataset = TransitionDataset(
        args.task, k=cfg["k"], scale=cfg["scale"], episodes=[ep], index=index
    )

    codes, steps, _ = stream_for_episode(model, dataset, ep.epis_idx, device)
    segments = sg.segments_from_track(meta.episode_labels(ep.epis_idx), ep.exec_start)

    # Streaming event log over the same stream.
    st = StreamingTokenizer(identity_span_encoder, CodeChangeBoundary(args.min_span))
    for c in codes:
        st.push(c)
    st.finish()

    canon = [sg.canonical_label(s.label) for s in segments]
    label_colours = _colour_map(canon)
    code_colours = _colour_map(codes + st.log.tokens)

    fig, axes = plt.subplots(3, 1, figsize=(13, 5.2), sharex=True)

    ax = axes[0]
    for x, c in zip(steps, codes):
        ax.axvspan(x, x + cfg["k"] / cfg["k"], color=code_colours[c], lw=0)
    ax.set_ylabel("raw\ncodes", rotation=0, ha="right", va="center")
    ax.set_yticks([])
    ax.set_title(
        f"{args.task} ep{ep.epis_idx}  N={ep.count}  "
        f"{len(codes)} transitions  k={cfg['k']}  "
        f"{len(set(codes))} distinct codes",
        fontsize=10,
    )

    ax = axes[1]
    for (lo, hi), tok in zip(st.log.spans, st.log.tokens):
        ax.axvspan(steps[lo], steps[min(hi, len(steps) - 1)], color=code_colours[tok], lw=0)
        ax.text(
            (steps[lo] + steps[min(hi, len(steps) - 1)]) / 2, 0.5, str(tok),
            ha="center", va="center", fontsize=7,
        )
    ax.set_ylabel("event\nlog", rotation=0, ha="right", va="center")
    ax.set_yticks([])

    ax = axes[2]
    for s, key in zip(segments, canon):
        ax.axvspan(s.start, s.end, color=label_colours[key], lw=0)
        ord_ = sg.ordinal(s.label)
        ax.text(
            (s.start + s.end) / 2, 0.5,
            key.replace("move to the top of the ", "").replace(" target", "")
            + (f" #{ord_}" if ord_ else ""),
            ha="center", va="center", fontsize=6, rotation=0,
        )
    ax.set_ylabel("subgoal\n(truth)", rotation=0, ha="right", va="center")
    ax.set_yticks([])
    ax.set_xlabel("frame")

    counts = st.log.counts()
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
    fig.suptitle(
        "event log counts: " + ", ".join(f"code {t}x{n}" for t, n in top),
        y=0.02, fontsize=8,
    )
    fig.tight_layout()

    out = args.out or str(
        paths.CACHE_ROOT / "figs" / f"timeline_{args.task}_ep{ep.epis_idx}.png"
    )
    import pathlib

    p = pathlib.Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150)
    print("wrote", p)

    # Text summary, so this is usable without opening the image.
    print(f"\nsubgoal events ({len(segments)}):")
    for s in segments:
        print(f"  t={s.start:4d}..{s.end:4d}  {s.label}")
    print(f"\nevent log ({len(st.log.tokens)} tokens): {st.log.tokens}")
    print(f"counts: {counts}")
    print(f"run-length: {st.log.run_length_encode()}")


if __name__ == "__main__":
    main()
