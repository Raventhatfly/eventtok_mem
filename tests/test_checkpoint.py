"""Resumable-checkpoint behaviour for preemptible partitions.

These matter because a run on ``kempner_requeue`` will be preempted, and a resume
that silently loses optimizer state — or that cannot read its own checkpoint —
wastes exactly the credits the cheap partition was supposed to save.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import torch
from torch import nn

from eventtok.train import checkpoint as ckpt


def _fixture():
    model = nn.Linear(4, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    return model, opt, sched


def _step(model, opt, sched, n=3):
    for _ in range(n):
        loss = model(torch.randn(8, 4)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()


def test_round_trip_restores_weights_optimizer_and_schedule(tmp_path) -> None:
    model, opt, sched = _fixture()
    _step(model, opt, sched, 5)
    state = ckpt.TrainState(epoch=2, step_in_epoch=7, global_step=57, seed=3)
    path = ckpt.save(tmp_path / "c.pt", model, opt, sched, state, {"a": 1})

    model2, opt2, sched2 = _fixture()
    restored = ckpt.restore(ckpt.load(path), model2, opt2, sched2)

    for a, b in zip(model.state_dict().values(), model2.state_dict().values()):
        assert torch.equal(a, b)
    # Adam moments must come back, or the loss curve visibly steps on resume.
    assert opt2.state_dict()["state"], "optimizer state was empty after restore"
    assert sched2.state_dict()["last_epoch"] == sched.state_dict()["last_epoch"]
    assert (restored.epoch, restored.step_in_epoch, restored.global_step) == (2, 7, 57)


def test_previous_checkpoint_is_kept_as_fallback(tmp_path) -> None:
    model, opt, sched = _fixture()
    path = tmp_path / "c.pt"
    ckpt.save(path, model, opt, sched, ckpt.TrainState(epoch=1), {})
    ckpt.save(path, model, opt, sched, ckpt.TrainState(epoch=2), {})
    assert path.with_name("c_prev.pt").is_file()


def test_truncated_checkpoint_falls_back_instead_of_failing(tmp_path) -> None:
    """Preemption mid-write must cost one interval, not the whole run."""
    model, opt, sched = _fixture()
    path = tmp_path / "c.pt"
    ckpt.save(path, model, opt, sched, ckpt.TrainState(epoch=1), {})
    ckpt.save(path, model, opt, sched, ckpt.TrainState(epoch=2), {})

    with open(path, "r+b") as fh:      # simulate a half-written file
        fh.truncate(64)

    blob = ckpt.load(path)
    assert blob is not None, "no fallback used; the run would restart from zero"
    assert ckpt.TrainState.from_dict(blob["train_state"]).epoch == 1


def test_no_checkpoint_returns_none(tmp_path) -> None:
    assert ckpt.load(tmp_path / "missing.pt") is None


def test_rng_state_restores_so_resume_is_reproducible(tmp_path) -> None:
    model, opt, sched = _fixture()
    torch.manual_seed(1234)
    path = ckpt.save(tmp_path / "c.pt", model, opt, sched, ckpt.TrainState(), {})
    expected = torch.randn(5)

    torch.manual_seed(999)                       # clobber it
    ckpt.restore(ckpt.load(path), model, opt, sched)
    assert torch.allclose(torch.randn(5), expected)


def test_handler_sets_flag_without_raising() -> None:
    """The flag is polled at step boundaries, so the signal must not interrupt."""
    handler = ckpt.PreemptionHandler().install()
    assert not handler.should_stop
    os.kill(os.getpid(), signal.SIGUSR1)
    assert handler.should_stop
    assert handler.signal_name == "SIGUSR1"
    signal.signal(signal.SIGUSR1, signal.SIG_DFL)


def test_trainer_exits_with_requeue_code_on_sigusr1(tmp_path) -> None:
    """End-to-end: a signalled process checkpoints and exits 85, not 0 or 1."""
    ckpt_path = tmp_path / "sig.pt"
    script = tmp_path / "run.py"
    script.write_text(
        "import os, signal, sys, torch\n"
        "from torch import nn\n"
        "from eventtok.train import checkpoint as ckpt\n"
        "m = nn.Linear(4, 3)\n"
        "o = torch.optim.AdamW(m.parameters())\n"
        "s = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=5)\n"
        "h = ckpt.PreemptionHandler().install()\n"
        "st = ckpt.TrainState()\n"
        "os.kill(os.getpid(), signal.SIGUSR1)\n"
        "for i in range(100):\n"
        "    st.global_step = i\n"
        "    if h.should_stop:\n"
        f"        ckpt.save({str(ckpt_path)!r}, m, o, s, st, {{}})\n"
        "        sys.exit(ckpt.REQUEUE_EXIT_CODE)\n"
        "sys.exit(0)\n"
    )
    env = dict(os.environ, PYTHONPATH=os.getcwd())
    proc = subprocess.run([sys.executable, str(script)], env=env, capture_output=True)
    assert proc.returncode == ckpt.REQUEUE_EXIT_CODE, proc.stderr.decode()[-400:]
    assert ckpt_path.is_file(), "exited without checkpointing"
