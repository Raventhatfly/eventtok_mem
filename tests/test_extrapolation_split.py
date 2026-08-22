"""The extrapolation split must not leak. If it does, the headline result is void.

The claim the experiment supports is "counts never seen in training are counted
correctly". That claim dies silently if any N in {4,5} episode reaches the k-means
fit, the BPE corpus, or the pattern search -- and a leak would *raise* the number,
so nothing in the output would look wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from eventtok import paths
from eventtok.data.index import RoboMMEIndex
from eventtok.eval import counting as cnt


@pytest.fixture(scope="module")
def index() -> RoboMMEIndex:
    paths.check_root()
    return RoboMMEIndex()


def test_by_count_partitions_cleanly(index: RoboMMEIndex) -> None:
    in_dist = index.by_count("PickXtimes", [1, 2, 3])
    ood = index.by_count("PickXtimes", [4, 5])
    assert len(in_dist) == 75 and len(ood) == 25
    assert not ({e.epis_idx for e in in_dist} & {e.epis_idx for e in ood})
    assert all(e.count <= 3 for e in in_dist)
    assert all(e.count >= 4 for e in ood)


def test_fit_split_excludes_every_ood_episode(index: RoboMMEIndex) -> None:
    """Reproduce the script's split and assert the fit set is clean."""
    in_dist = index.by_count("PickXtimes", [1, 2, 3])
    ood_ids = {e.epis_idx for e in index.by_count("PickXtimes", [4, 5])}

    order = np.random.default_rng(0).permutation(len(in_dist))
    fit_ids = {in_dist[i].epis_idx for i in order[: len(in_dist) // 2]}
    id_test_ids = {in_dist[i].epis_idx for i in order[len(in_dist) // 2 :]}

    assert not (fit_ids & ood_ids), "OOD episode leaked into the k-means/pattern fit"
    assert not (fit_ids & id_test_ids), "fit and in-distribution test overlap"
    assert fit_ids | id_test_ids == {e.epis_idx for e in in_dist}


def test_select_pattern_only_reads_the_ids_it_is_given() -> None:
    """A pattern chosen on a subset must not change when unseen episodes are added.

    This is the guard that makes the split meaningful: select_pattern takes explicit
    ids, and episodes outside them must be invisible to it even though they are
    present in the runs/counts dicts it receives.
    """
    runs = {
        1: [7, 3, 7, 3],
        2: [7, 3, 7, 3, 7, 3],
        3: [7, 3],
        # An "unseen" episode with a very different structure and a large N.
        99: [5, 5, 5, 5, 5, 5, 5, 5, 5],
    }
    counts = {1: 2, 2: 3, 3: 1, 99: 9}
    train_ids = [1, 2, 3]

    chosen = cnt.select_pattern(runs, counts, train_ids, max_len=3)
    assert chosen is not None

    # Same call, with the unseen episode removed entirely.
    lean = cnt.select_pattern(
        {k: v for k, v in runs.items() if k in train_ids},
        {k: v for k, v in counts.items() if k in train_ids},
        train_ids,
        max_len=3,
    )
    assert lean is not None
    assert chosen.gram == lean.gram
    assert chosen.train_accuracy == lean.train_accuracy


def test_shuffled_baseline_is_high_when_few_labels_occur() -> None:
    """On the OOD set only N in {4,5} occurs, so chance is ~50%, not 0%.

    Recorded as a test because an OOD accuracy of 55% would look like a result
    against an assumed-zero baseline and is in fact nothing.
    """
    runs = {i: [1] * (4 if i < 12 else 5) for i in range(25)}
    counts = {i: (4 if i < 12 else 5) for i in range(25)}
    pattern = cnt.CountPattern(gram=(1,), length=1, train_accuracy=1.0,
                               train_correlation=1.0)
    mean, _ = cnt.shuffled_baseline(runs, counts, pattern, list(range(25)), trials=200)
    assert 0.4 < mean < 0.6
