"""Resumable checkpoints for preemptible (requeue) partitions.

``kempner_requeue`` is cheap because jobs get preempted, so a run must survive
being killed at an arbitrary moment and pick up where it stopped. That needs three
things, and missing any one of them silently wastes the credits the partition was
supposed to save:

1. **Everything that defines training state**, not just weights — optimizer
   moments, LR-scheduler position, epoch *and* step within epoch, and RNG state.
   Resuming with a fresh optimizer throws away Adam's moment estimates and puts a
   visible discontinuity in the loss curve.
2. **Atomic writes.** Preemption during ``torch.save`` leaves a truncated file. A
   run that dies at hour 20 and then cannot load its own checkpoint is worse than
   one that never checkpointed. Write to a temp path, ``fsync``, then rename —
   rename is atomic within a filesystem.
3. **Two files, not one.** ``last.pt`` is overwritten constantly; ``last_prev.pt``
   is the previous good one. If the newest is corrupt for any reason, the run falls
   back one interval instead of to zero.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Convention already used elsewhere in these repos: the trainer exits with this
# code to mean "clean checkpoint boundary reached, please requeue me".
REQUEUE_EXIT_CODE = int(os.environ.get("REQUEUE_EXIT_CODE", "85"))


@dataclass
class TrainState:
    """Everything needed to resume mid-run."""

    epoch: int = 0
    step_in_epoch: int = 0
    global_step: int = 0
    history: list[dict] = field(default_factory=list)
    seed: int = 0

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "step_in_epoch": self.step_in_epoch,
            "global_step": self.global_step,
            "history": self.history,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainState":
        return cls(
            epoch=int(d.get("epoch", 0)),
            step_in_epoch=int(d.get("step_in_epoch", 0)),
            global_step=int(d.get("global_step", 0)),
            history=list(d.get("history", [])),
            seed=int(d.get("seed", 0)),
        )


def _rng_state() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng(state: dict) -> None:
    if not state:
        return
    torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        except Exception:
            pass  # different GPU count on requeue; not worth failing over
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def save(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: TrainState,
    args: dict,
    keep_prev: bool = True,
) -> Path:
    """Atomically write a resumable checkpoint. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    blob = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "train_state": state.to_dict(),
        "rng": _rng_state(),
        "args": args,
        # Kept for compatibility with load_model(), which reads these two.
        "history": state.history,
    }

    if keep_prev and path.is_file():
        prev = path.with_name(path.stem + "_prev" + path.suffix)
        try:
            os.replace(path, prev)
        except OSError:
            pass

    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(blob, fh)
        fh.flush()
        os.fsync(fh.fileno())      # survive a node dying, not just the process
    os.replace(tmp, path)          # atomic within one filesystem
    return path


def load(path: Path, map_location="cpu") -> dict | None:
    """Load a checkpoint, falling back to the previous one if the newest is bad."""
    path = Path(path)
    candidates = [path, path.with_name(path.stem + "_prev" + path.suffix)]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            blob = torch.load(candidate, map_location=map_location, weights_only=False)
        except Exception as exc:                       # truncated by preemption
            print(f"[ckpt] {candidate.name} unreadable ({exc}); trying older", flush=True)
            continue
        if candidate != path:
            print(f"[ckpt] resumed from fallback {candidate.name}", flush=True)
        return blob
    return None


def restore(
    blob: dict,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
) -> TrainState:
    """Put a loaded checkpoint back into the objects. Returns the train state."""
    model.load_state_dict(blob["state_dict"])
    if optimizer is not None and blob.get("optimizer") is not None:
        optimizer.load_state_dict(blob["optimizer"])
    if scheduler is not None and blob.get("scheduler") is not None:
        scheduler.load_state_dict(blob["scheduler"])
    _restore_rng(blob.get("rng", {}))
    return TrainState.from_dict(blob.get("train_state", {}))


class PreemptionHandler:
    """Catches the pre-preemption signal and asks the trainer to stop cleanly.

    SLURM's ``--signal=B:USR1@180`` notifies the batch shell 180 s before the job
    is killed; the shell forwards it here. SIGTERM is also caught because
    preemption itself arrives as TERM followed by KILL.

    The flag is checked at step boundaries rather than acting immediately, so a
    checkpoint is never written from the middle of a backward pass.
    """

    def __init__(self) -> None:
        self.should_stop = False
        self.signal_name: str | None = None

    def install(self) -> "PreemptionHandler":
        import signal

        def handler(signum, _frame):
            self.should_stop = True
            self.signal_name = signal.Signals(signum).name
            print(
                f"[preempt] caught {self.signal_name}; will checkpoint and requeue "
                f"at the next step boundary",
                flush=True,
            )

        for sig in (signal.SIGUSR1, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass  # not the main thread, or unsupported platform
        return self
