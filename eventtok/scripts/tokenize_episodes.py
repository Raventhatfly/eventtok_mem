"""Run a trained tokenizer over whole episodes and dump the code streams.

    python -m eventtok.scripts.tokenize_episodes --ckpt ... --task SwingXtimes

Writes ``{epis_idx: [code, code, ...]}`` in temporal order, which is the input to
the BPE stage and to the timeline plot. Also reports codebook usage over the
*whole* task rather than over one batch, which is the only honest number.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .. import paths
from ..data.index import RoboMMEIndex
from ..data.robomme import TransitionDataset
from ..models.tokenizer import EventTokenizer


def load_model(ckpt_path: str, device: torch.device) -> tuple[EventTokenizer, dict]:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = blob["args"]
    model = EventTokenizer(
        action_dim=8,
        k=a["k"],
        d_feat=2048,
        n_vis_tokens=4 if a["scale"] == "2x2" else (16 if a["scale"] == "4x4" else 64),
        d_model=a["d_model"],
        n_registers=a["registers"],
        n_layers=a["layers"],
        fsq_levels=tuple(a["levels"]),
        causal_registers=not a["no_causal"],
        use_vision=a.get("use_vision", False),
        nested_dropout=a.get("nested_dropout", False),
        far_head=a.get("far_horizon") is not None,
    ).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, a


@torch.no_grad()
def stream_for_episode(
    model: EventTokenizer,
    dataset: TransitionDataset,
    epis_idx: int,
    device: torch.device,
    batch: int = 256,
) -> tuple[list[int], list[int], np.ndarray]:
    """-> (code ids, transition start frames, per-register digits)."""
    positions = dataset.episode_transitions(epis_idx)
    codes: list[int] = []
    steps: list[int] = []
    digits: list[np.ndarray] = []
    for lo in range(0, len(positions), batch):
        items = [dataset[p] for p in positions[lo : lo + batch]]
        feat_t = torch.stack([i["feat_t"] for i in items]).to(device)
        feat_next = torch.stack([i["feat_next"] for i in items]).to(device)
        actions = torch.stack([i["actions"] for i in items]).to(device)
        out = model(actions, feat_t, feat_next)
        # Register 0 is the coarse token under the causal ordering, so it is the
        # one expected to repeat across instances of the same event. Later
        # registers carry instance detail and are kept only in `digits`.
        codes.extend(out.tokens[:, 0].cpu().tolist())
        digits.append(out.digits.cpu().numpy())
        steps.extend(int(i["step"]) for i in items)
    return codes, steps, np.concatenate(digits) if digits else np.zeros((0, 0, 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="SwingXtimes")
    ap.add_argument("--limit-episodes", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    paths.check_root()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = load_model(args.ckpt, device)

    index = RoboMMEIndex()
    episodes = index.by_task(args.task)
    if args.limit_episodes:
        episodes = episodes[: args.limit_episodes]
    dataset = TransitionDataset(
        args.task, k=cfg["k"], scale=cfg["scale"], episodes=episodes, index=index
    )
    print(f"{args.task}: {len(episodes)} episodes, {len(dataset)} transitions", flush=True)

    streams: dict[int, dict] = {}
    all_codes: list[int] = []
    all_digits: list[np.ndarray] = []
    for i, ep in enumerate(episodes):
        codes, steps, digits = stream_for_episode(model, dataset, ep.epis_idx, device)
        streams[ep.epis_idx] = {
            "codes": codes,
            "steps": steps,
            "count": ep.count,
            "prompt": ep.prompt,
        }
        all_codes.extend(codes)
        if digits.size:
            all_digits.append(digits)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(episodes)} episodes", flush=True)

    codes_arr = np.asarray(all_codes)
    uniq, counts = np.unique(codes_arr, return_counts=True)
    top = np.argsort(-counts)[:12]
    print(
        f"\ncodebook usage over the whole task: {len(uniq)}/{model.codebook_size} "
        f"({len(uniq) / model.codebook_size:.1%}) across {len(codes_arr)} transitions"
    )
    print("most frequent codes:", [(int(uniq[j]), int(counts[j])) for j in top])

    # Effective vocabulary: a dead FSQ channel is invisible in the aggregate
    # number above, so report the per-channel occupancy too.
    if all_digits:
        digits = np.concatenate(all_digits)[:, 0, :]  # register 0
        eff = 1
        per = []
        for c in range(digits.shape[1]):
            used = len(np.unique(digits[:, c]))
            per.append(f"{used}/{model.fsq.levels[c]}")
            eff *= used
        print(f"per-channel levels used: {' '.join(per)}  -> effective vocab {eff}")

    out_path = args.out or str(paths.CACHE_ROOT / "streams" / f"{args.task}.json")
    import pathlib

    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        json.dump({str(k): v for k, v in streams.items()}, fh)
    print("wrote", p)


if __name__ == "__main__":
    main()
