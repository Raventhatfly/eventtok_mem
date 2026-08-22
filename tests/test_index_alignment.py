"""M0 exit criteria. These pin facts about the on-disk layout that the rest of
the pipeline assumes; if any fail, the data root is wrong or the dataset changed.

Run: python -m pytest tests/ -q
"""

from __future__ import annotations

import numpy as np
import pytest

from eventtok import paths
from eventtok.data import prompts, subgoals as sg
from eventtok.data.index import EPISODES_PER_TASK, TASKS, RoboMMEIndex


@pytest.fixture(scope="module")
def index() -> RoboMMEIndex:
    paths.check_root()
    return RoboMMEIndex()


def test_shape(index: RoboMMEIndex) -> None:
    assert len(index) == 1600
    assert len(index.pairs) == 476_857
    assert len(TASKS) == 16


def test_task_from_epis_idx(index: RoboMMEIndex) -> None:
    """epis_idx // 100 indexes the alphabetical task list, for all 1600."""
    for ep in index.episodes:
        assert ep.task == TASKS[ep.epis_idx // EPISODES_PER_TASK]
        assert ep.episode_id == ep.epis_idx % EPISODES_PER_TASK


def test_pkl_id_round_trip(index: RoboMMEIndex) -> None:
    """locate() and Episode.pkl_id() are inverses."""
    rng = np.random.default_rng(0)
    for pkl_id in rng.integers(0, len(index.pairs), size=3000):
        epis_idx, step_idx = index.locate(int(pkl_id))
        assert index[epis_idx].pkl_id(step_idx) == pkl_id


def test_pkl_ranges_are_contiguous(index: RoboMMEIndex) -> None:
    """data/ is a gapless concatenation of execution frames in episode order."""
    expected_lo = 0
    for ep in index.episodes:
        assert ep.pkl_lo == expected_lo
        expected_lo = ep.pkl_hi + 1
    assert expected_lo == len(index.pairs)


def test_exec_start_is_not_always_zero(index: RoboMMEIndex) -> None:
    """900 of 1600 episodes have a pre-execution prefix.

    Guards against reintroducing the assumption that step_idx starts at 0.
    """
    with_prefix = sum(1 for ep in index.episodes if ep.exec_start > 0)
    assert with_prefix == 900


def test_counting_task_distributions(index: RoboMMEIndex) -> None:
    """N comes only from the prompt, and SwingXtimes tops out at 3.

    The latter is why the extrapolation experiment must use PickXtimes.
    """
    swing = [ep.count for ep in index.by_task("SwingXtimes")]
    pick = [ep.count for ep in index.by_task("PickXtimes")]
    assert {c: swing.count(c) for c in sorted(set(swing))} == {1: 23, 2: 31, 3: 46}
    assert {c: pick.count(c) for c in sorted(set(pick))} == {
        1: 23, 2: 25, 3: 27, 4: 12, 5: 13,
    }
    assert max(swing) == 3, "SwingXtimes cannot test N >= 5"
    assert len(index.by_count("PickXtimes", [4, 5])) == 25


def test_non_counting_task_has_no_count(index: RoboMMEIndex) -> None:
    assert index[235].count is None
    assert not prompts.has_count("ButtonUnmaskSwap")


def test_episode_235_boundaries(index: RoboMMEIndex) -> None:
    """The reference timeline. Reads pkls directly, so it is slow but exact."""
    ep = index[235]
    assert ep.task == "ButtonUnmaskSwap"
    segments = sg.segments_from_track(sg.read_subgoal_track(ep), ep.exec_start)
    assert sg.boundaries(segments) == [0, 117, 213, 322, 363]
    assert segments[0].label == "press the first button"
    assert segments[-1].end == ep.n_frames


def test_ordinal_canonicalisation() -> None:
    """Repeats must collapse to one label, or code-consistency scoring is wrong."""
    a = "move to the top of the right-side target for the first time"
    b = "move to the top of the right-side target for the second time"
    assert sg.canonical_label(a) == sg.canonical_label(b)
    assert sg.canonical_label(a) == "move to the top of the right-side target"
    assert (sg.ordinal(a), sg.ordinal(b)) == (1, 2)
    assert sg.ordinal("press the button") is None
    assert sg.canonical_label("press the button") == "press the button"


def test_observable_label_collapses_object_identity() -> None:
    """Object identity must be collapsed before scoring per-transition codes.

    On occlusion tasks the named object is hidden at the moment of the action and
    varies only across episodes, so no code computed from one transition can carry
    it. Scoring against the raw label measures an impossible target -- it read
    38.0% instead of 53.9% on ButtonUnmask and inverted the conclusion about
    whether vision helps.
    """
    red = "pick up the container that hides the red cube"
    green = "pick up the container that hides the green cube"
    assert sg.observable_label(red) == sg.observable_label(green)
    assert "<obj>" in sg.observable_label(red)
    # Distinct actions must stay distinct.
    assert sg.observable_label(red) != sg.observable_label("press the button")
    # And the ordinal stripping still composes with it.
    a = "move to the top of the right-side target for the first time"
    b = "move to the top of the right-side target for the second time"
    assert sg.observable_label(a) == sg.observable_label(b)
