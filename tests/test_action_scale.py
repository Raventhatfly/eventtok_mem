"""Dead action dimensions must not be amplified. This one collapsed two tokenizers.

The gripper channel breaks normalisation in both directions. On grasping tasks its std
is ~0.95 against 0.05-0.20 for the pose dimensions, so leaving it unnormalised makes
the loss a gripper predictor. On tasks with no grasping it never actuates, its std is
~0, and flooring the divisor turns normalisation into a 1000x amplifier of a constant
channel -- which is what collapsed the learned tokenizer to 1 of 64 codes on
PatternLock and RouteStick, the only two collapses across 15 tasks.
"""

from __future__ import annotations

import numpy as np
import pytest

from eventtok import paths
from eventtok.data.meta import TaskMeta


def _max_normalised(meta: TaskMeta, n: int = 512) -> float:
    scale = meta.action_scale
    rows = np.linspace(0, len(meta.actions) - 1, n).astype(int)
    chunks = np.stack([meta.delta_actions(int(r)) / scale for r in rows])
    return float(np.abs(chunks).max())


def test_dead_dimensions_are_zeroed_not_amplified() -> None:
    """PatternLock's gripper is constant, so it must contribute exactly nothing."""
    paths.check_root()
    meta = TaskMeta("PatternLock")
    assert 7 in meta.dead_action_dims, "gripper should be dead on a non-grasping task"
    assert not np.isfinite(meta.action_scale[7])

    chunk = meta.delta_actions(0) / meta.action_scale
    assert np.all(chunk[:, 7] == 0.0), "a constant channel must normalise to zero"
    assert np.isfinite(chunk).all(), "no NaN or inf may leak into the features"


def test_grasping_tasks_keep_their_gripper() -> None:
    """The fix must not zero a gripper that actually moves."""
    paths.check_root()
    meta = TaskMeta("PickXtimes")
    assert len(meta.dead_action_dims) == 0
    assert np.isfinite(meta.action_scale).all()
    assert meta.action_scale[7] > 0.5


@pytest.mark.parametrize("task", ["PatternLock", "RouteStick", "SwingXtimes", "BinFill"])
def test_normalised_actions_stay_in_a_sane_range(task: str) -> None:
    """Every task must land in the same order of magnitude after normalisation.

    Before the fix PatternLock and RouteStick hit 1000 while the rest sat at 8-12,
    and nothing in the pipeline noticed until a training run collapsed.
    """
    paths.check_root()
    assert _max_normalised(TaskMeta(task)) < 50.0
