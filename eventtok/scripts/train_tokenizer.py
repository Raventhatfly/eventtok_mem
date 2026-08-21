"""Train the event tokenizer on one task.

    python -m eventtok.scripts.train_tokenizer --task SwingXtimes --epochs 4

Logs both head losses separately, plus codebook usage and the per-channel FSQ
level histogram. The per-channel histogram is the one that matters: FSQ cannot
have dead codes the way VQ does, but a channel collapsing onto one or two levels
costs a factor of L_i of effective vocabulary while aggregate usage still looks
fine.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch.utils.data import DataLoader

from .. import paths
from ..data.index import RoboMMEIndex
from ..data.robomme import TransitionDataset
from ..models.tokenizer import EventTokenizer, losses


def build(args, dataset: TransitionDataset) -> EventTokenizer:
    sample = dataset[0]
    return EventTokenizer(
        d_feat=sample["feat_t"].shape[-1],
        n_vis_tokens=sample["feat_t"].shape[0],
        action_dim=sample["actions"].shape[-1],
        k=args.k,
        d_model=args.d_model,
        n_registers=args.registers,
        n_layers=args.layers,
        fsq_levels=tuple(args.levels),
        causal_registers=not args.no_causal,
    )


def code_stats(model: EventTokenizer, tokens: torch.Tensor, digits: torch.Tensor) -> dict:
    used = int(torch.unique(tokens).numel())
    hist = model.fsq.level_histogram(digits)
    # Fraction of each channel's levels that ever fire, and the entropy of the
    # channel marginal in nats — a channel pinned to one level has entropy 0.
    per_channel = []
    for h in hist:
        p = h.float() / h.sum().clamp(min=1)
        nz = p[p > 0]
        per_channel.append(
            {
                "levels_used": int((h > 0).sum()),
                "levels": int(h.numel()),
                "entropy": float(-(nz * nz.log()).sum()),
            }
        )
    return {
        "codes_used": used,
        "codebook": model.codebook_size,
        "usage": used / model.codebook_size,
        "channels": per_channel,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="SwingXtimes")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--scale", default="2x2")
    ap.add_argument("--levels", type=int, nargs="+", default=[8, 8, 8])
    ap.add_argument("--registers", type=int, default=2)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--no-causal", action="store_true")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w-feat", type=float, default=1.0)
    ap.add_argument("--w-action", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit-episodes", type=int, default=None)
    ap.add_argument("--out", default=None, help="checkpoint path")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    paths.check_root()
    index = RoboMMEIndex()
    episodes = index.by_task(args.task)
    if args.limit_episodes:
        episodes = episodes[: args.limit_episodes]

    dataset = TransitionDataset(
        args.task, k=args.k, scale=args.scale, stride=args.stride,
        episodes=episodes, index=index,
    )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(
        f"{args.task}: {len(episodes)} episodes, {len(dataset)} transitions, "
        f"k={args.k}, device={device}",
        flush=True,
    )

    model = build(args, dataset).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"model: {n_params/1e6:.2f}M params, codebook={model.codebook_size} "
        f"levels={tuple(args.levels)} registers={args.registers}",
        flush=True,
    )

    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, drop_last=True, pin_memory=(device.type == "cuda"),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs * len(loader))
    )

    history = []
    for epoch in range(args.epochs):
        model.train()
        agg = {"feat": 0.0, "action": 0.0, "total": 0.0, "residual_cos": 0.0}
        n = 0
        last_tokens = last_digits = None
        t0 = time.time()
        for step, batch in enumerate(loader):
            feat_t = batch["feat_t"].to(device, non_blocking=True)
            feat_next = batch["feat_next"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)

            out = model(feat_t, feat_next, actions)
            loss = losses(out, feat_t, feat_next, actions, args.w_feat, args.w_action)

            opt.zero_grad(set_to_none=True)
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            for key in agg:
                agg[key] += float(loss[key].detach())
            n += 1
            last_tokens, last_digits = out.tokens.detach(), out.digits.detach()

            if step % 50 == 0:
                print(
                    f"  ep{epoch} step {step:5d}/{len(loader)} "
                    f"feat {float(loss['feat']):.4f} action {float(loss['action']):.4f} "
                    f"rcos {float(loss['residual_cos']):+.4f}",
                    flush=True,
                )

        stats = code_stats(model, last_tokens, last_digits)
        record = {
            "epoch": epoch,
            "feat": agg["feat"] / n,
            "action": agg["action"] / n,
            "total": agg["total"] / n,
            "residual_cos": agg["residual_cos"] / n,
            "secs": time.time() - t0,
            **stats,
        }
        history.append(record)
        chans = " ".join(
            f"{c['levels_used']}/{c['levels']}(H={c['entropy']:.2f})"
            for c in stats["channels"]
        )
        print(
            f"[epoch {epoch}] feat {record['feat']:.4f} action {record['action']:.4f} "
            f"rcos {record['residual_cos']:+.4f} "
            f"| codes {stats['codes_used']}/{stats['codebook']} "
            f"({stats['usage']:.1%}) | channels {chans} | {record['secs']:.0f}s",
            flush=True,
        )

    out_path = args.out or str(
        paths.CACHE_ROOT / "ckpt" / f"tokenizer_{args.task}_{'x'.join(map(str, args.levels))}.pt"
    )
    import pathlib

    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args),
            "history": history,
        },
        p,
    )
    print("saved", p, flush=True)
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
