"""Train the event tokenizer on one task.

    python -m eventtok.scripts.train_tokenizer --task SwingXtimes --epochs 4

Built for preemptible partitions (``kempner_requeue``): checkpoints every
``--save-every`` steps, resumes mid-epoch by default, and on SIGUSR1/SIGTERM saves
at the next step boundary and exits with ``REQUEUE_EXIT_CODE`` (85) so the batch
script can requeue itself.

Logs both head losses separately, plus codebook usage and the per-channel FSQ
level histogram. The per-channel histogram is the one that matters: FSQ cannot
have dead codes the way VQ does, but a channel collapsing onto one or two levels
costs a factor of L_i of effective vocabulary while aggregate usage still looks
fine.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch
from torch.utils.data import DataLoader

from .. import paths
from ..train import checkpoint as ckpt
from ..data.index import RoboMMEIndex
from ..data.robomme import TransitionDataset
from ..models.tokenizer import EventTokenizer, losses


def build(args, dataset: TransitionDataset) -> EventTokenizer:
    sample = dataset[0]
    return EventTokenizer(
        action_dim=sample["actions"].shape[-1],
        d_feat=sample["feat_t"].shape[-1],
        n_vis_tokens=sample["feat_t"].shape[0],
        k=args.k,
        d_model=args.d_model,
        n_registers=args.registers,
        n_layers=args.layers,
        fsq_levels=tuple(args.levels),
        causal_registers=not args.no_causal,
        use_vision=args.use_vision,
        far_head=args.far_horizon is not None,
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
    ap.add_argument("--use-vision", action="store_true",
                    help="add visual context to the encoder (off by default: it "
                         "pulled codes toward trajectory phase, not motion type)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w-action", type=float, default=1.0)
    ap.add_argument("--w-far", type=float, default=0.0)
    ap.add_argument("--far-horizon", type=int, default=None,
                    help="auxiliary target frame at t+far_horizon (must exceed k)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit-episodes", type=int, default=None)
    ap.add_argument("--out", default=None, help="checkpoint path")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=200,
                    help="checkpoint every N optimizer steps (0 = epoch ends only)")
    ap.add_argument("--resume", default="auto", choices=["auto", "never"],
                    help="auto resumes from --out if it exists")
    args = ap.parse_args()

    paths.check_root()
    index = RoboMMEIndex()
    episodes = index.by_task(args.task)
    if args.limit_episodes:
        episodes = episodes[: args.limit_episodes]

    dataset = TransitionDataset(
        args.task, k=args.k, scale=args.scale, stride=args.stride,
        episodes=episodes, index=index, far_horizon=args.far_horizon,
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

    # A per-epoch seeded generator makes the batch order reproducible, which is
    # what lets a mid-epoch resume skip forward to the exact same position.
    gen = torch.Generator()
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, generator=gen,
        num_workers=args.workers, drop_last=True, pin_memory=(device.type == "cuda"),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs * len(loader))
    )

    out_path = pathlib.Path(
        args.out
        or paths.CACHE_ROOT / "ckpt" / f"tokenizer_{args.task}_{'x'.join(map(str, args.levels))}.pt"
    )

    torch.manual_seed(args.seed)
    state = ckpt.TrainState(seed=args.seed)
    if args.resume == "auto":
        blob = ckpt.load(out_path, map_location=device)
        if blob is not None:
            state = ckpt.restore(blob, model, opt, sched)
            print(
                f"[resume] epoch {state.epoch}, step_in_epoch {state.step_in_epoch}, "
                f"global_step {state.global_step}",
                flush=True,
            )

    preempt = ckpt.PreemptionHandler().install()

    def checkpoint_now() -> None:
        ckpt.save(out_path, model, opt, sched, state, vars(args))

    history = state.history
    # Captured before the loop: state.epoch is reassigned each iteration, so
    # comparing against it inside the loop would make the skip always fire.
    resume_epoch, resume_step = state.epoch, state.step_in_epoch
    for epoch in range(resume_epoch, args.epochs):
        state.epoch = epoch
        gen.manual_seed(args.seed * 100003 + epoch)
        model.train()
        agg = {"action": 0.0, "total": 0.0}
        n = 0
        last_tokens = last_digits = None
        t0 = time.time()
        # Resuming mid-epoch: the sampler is reseeded per epoch, so the batch
        # order is reproducible and skipping restores the exact position.
        skip = resume_step if epoch == resume_epoch else 0
        if skip:
            print(f"  [resume] skipping {skip} batches of epoch {epoch}", flush=True)
        for step, batch in enumerate(loader):
            if step < skip:
                continue
            feat_t = batch["feat_t"].to(device, non_blocking=True)
            feat_next = batch["feat_next"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            feat_far = (
                batch["feat_far"].to(device, non_blocking=True)
                if "feat_far" in batch else None
            )

            out = model(actions, feat_t, feat_next, feat_far)
            loss = losses(out, actions, feat_t, feat_far, args.w_action, args.w_far)

            opt.zero_grad(set_to_none=True)
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            for key in agg:
                agg[key] += float(loss[key].detach())
            if "far_cos" in loss:
                agg["far_cos"] = agg.get("far_cos", 0.0) + float(loss["far_cos"])
            n += 1
            state.step_in_epoch = step + 1
            state.global_step += 1
            last_tokens, last_digits = out.tokens.detach(), out.digits.detach()

            if args.save_every and state.global_step % args.save_every == 0:
                checkpoint_now()

            if preempt.should_stop:
                checkpoint_now()
                print(
                    f"[preempt] checkpointed at epoch {epoch} step {step + 1}; "
                    f"exiting {ckpt.REQUEUE_EXIT_CODE} to request requeue",
                    flush=True,
                )
                sys.exit(ckpt.REQUEUE_EXIT_CODE)

            if step % 50 == 0:
                extra = (
                    f" far_cos {float(loss['far_cos']):+.4f}"
                    if "far_cos" in loss else ""
                )
                print(
                    f"  ep{epoch} step {step:5d}/{len(loader)} "
                    f"action {float(loss['action']):.4f}{extra}",
                    flush=True,
                )

        stats = code_stats(model, last_tokens, last_digits)
        record = {
            "epoch": epoch,
            **{k2: v / n for k2, v in agg.items()},
            "secs": time.time() - t0,
            **stats,
        }
        history.append(record)
        state.history = history
        # Point at the *next* epoch so a resume does not redo this one. Mid-epoch
        # saves inside the loop above keep state.epoch == epoch on purpose.
        state.epoch = epoch + 1
        state.step_in_epoch = 0
        checkpoint_now()
        chans = " ".join(
            f"{c['levels_used']}/{c['levels']}(H={c['entropy']:.2f})"
            for c in stats["channels"]
        )
        print(
            f"[epoch {epoch}] action {record['action']:.4f} "
            f"{'far_cos %+.4f ' % record['far_cos'] if 'far_cos' in record else ''}"
            f"| codes {stats['codes_used']}/{stats['codebook']} "
            f"({stats['usage']:.1%}) | channels {chans} | {record['secs']:.0f}s",
            flush=True,
        )

    checkpoint_now()
    print("saved", out_path, flush=True)
    if history:
        print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
