"""The pi0.5 memory slot: encoding, frame indexing, and the controls.

No JAX here. Everything upstream of ``PerceptualMemory`` is plain numpy, and that is
where the mistakes this project has already made live -- a wrong frame offset, a log
that is a set rather than a sequence, a control that is not actually a control.
"""

from __future__ import annotations

import numpy as np
import pytest

from eventtok.pi05.config import history_config
from eventtok.pi05.dataset import EventMemoryMixin
from eventtok.pi05.features import EventMemoryFeatures, slot_posemb
from eventtok.pi05.tokens import EventLogCache, cache_path

TAG = "default"
TASK = "ButtonUnmask"
pytestmark = pytest.mark.skipif(
    not cache_path(TASK, TAG).is_file(), reason=f"no {TASK} event cache"
)


@pytest.fixture(scope="module")
def cache():
    return EventLogCache(TASK, TAG)


# ------------------------------------------------------------------ encoding
def test_onehot_is_an_embedding_lookup():
    f = EventMemoryFeatures(n_symbols=12, max_log=6)
    out = f(np.array([3, 3, 7, 0, 0, 0]), 3, np.zeros(12))
    img, mask = out["static_image_emb"], out["static_mask"]
    assert img[mask].sum(axis=1).tolist() == [1.0, 1.0, 1.0]
    assert img[~mask].sum() == 0.0
    assert img[0].argmax() == 3 and img[1].argmax() == 3 and img[2].argmax() == 7
    # The two 3s are identical rows: repeats must not be collapsed or perturbed,
    # since their multiplicity is the quantity the method claims to preserve.
    assert np.array_equal(img[0], img[1])


def test_slots_are_distinguishable():
    # A bidirectional prefix sees the memory as a set unless positions differ, and
    # "swung three times" then reads the same as "swung once".
    pos = slot_posemb(16, 32)
    assert len(np.unique(pos, axis=0)) == 16


def test_overflow_reaches_the_encoder():
    f = EventMemoryFeatures(n_symbols=5, max_log=4)
    ov = np.zeros(5)
    ov[2] = 9
    st = f(np.zeros(4, dtype=int), 0, ov)["static_state_emb"]
    assert st.shape == (4, 5)
    assert np.allclose(st[:, 2], np.log1p(9))
    assert np.allclose(np.delete(st, 2, axis=1), 0.0)


def test_out_of_vocabulary_token_raises():
    f = EventMemoryFeatures(n_symbols=4, max_log=3)
    with pytest.raises(ValueError, match="vocabulary"):
        f(np.array([9, 0, 0]), 1, np.zeros(4))


def test_config_dims_match_the_cache(cache):
    cfg = history_config(cache.n_symbols, cache.max_log)
    assert cfg["budget"] == cache.max_log
    assert cfg["memory_feature"]["img"]["input_dim"] == cache.n_symbols
    assert cfg["memory_feature"]["state"]["input_dim"] == cache.n_symbols
    assert cfg["use_pos_emb"] and cfg["use_state_emb"]
    assert cfg["representation_type"] == "perceptual"


# ------------------------------------------------------------------- dataset
class _FakeBase:
    """Stands in for RoboMMEDataset: the row keys the event dataset reads.

    Takes the real keyword signature so the subclass's ``super().__init__`` call is
    exercised as written -- including that it passes ``history_config=None``, which is
    what keeps the parent from loading per-frame features.
    """

    def __init__(self, dataset_path, data_config, history_config, action_horizon,
                 compute_norm_stats=False):
        assert history_config is None, "the parent must not run its own memory prep"
        self.rows = dataset_path

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return dict(self.rows[idx])


class _EventFake(EventMemoryMixin, _FakeBase):
    """Module level so it pickles, the way the real subclass has to."""


def _make(rows, **kwargs):
    return _EventFake(rows, None, None, 20, event_tag=TAG, **kwargs)


def test_frame_index_subtracts_exec_start(cache):
    # Pick an episode and offset where the two readings genuinely differ; roughly half
    # of all frames have an empty log, and an empty-vs-empty comparison would pass
    # whether or not the subtraction happens.
    offset = 40
    for ep in cache.meta["train_episodes"]:
        ep = int(ep)
        frames = sorted(t for (e, t) in cache._row if e == ep)
        hit = [t for t in frames if t + offset in frames
               and cache[(ep, t)][1] != cache[(ep, t + offset)][1]]
        if hit:
            break
    else:
        pytest.skip("no frame pair with differing logs")
    frame = hit[0]

    rows = [{"epis_idx": ep, "step_idx": frame + offset, "exec_start_idx": offset}]
    got = _make(rows)[0]
    f = EventMemoryFeatures(cache.n_symbols, cache.max_log)
    exp = f(*cache.get(ep, frame))            # step_idx - exec_start
    wrong = f(*cache.get(ep, frame + offset))  # what ignoring exec_start would give
    assert np.array_equal(got["static_image_emb"], exp["static_image_emb"])
    assert not np.array_equal(exp["static_mask"], wrong["static_mask"])


def test_blank_mode_empties_the_log(cache):
    ep = int(cache.meta["train_episodes"][0])
    rows = [{"epis_idx": ep, "step_idx": 200, "exec_start_idx": 0}]
    got = _make(rows, event_mode="blank")[0]
    assert not got["static_mask"].any()
    assert got["static_image_emb"].sum() == 0.0


def test_wrong_mode_swaps_the_episode(cache):
    ep = int(cache.meta["train_episodes"][0])
    rows = [{"epis_idx": ep, "step_idx": 250, "exec_start_idx": 0}]
    right = _make(rows)[0]["static_image_emb"]
    seen = {right.tobytes()}
    for seed in range(8):
        seen.add(_make(rows, event_mode="wrong", event_seed=seed)[0]
                 ["static_image_emb"].tobytes())
    # A control that returns the correct log would make every comparison against it
    # vacuous, so require that it actually differs at least once.
    assert len(seen) > 1


def test_drop_blanks_some_frames(cache):
    ep = int(cache.meta["train_episodes"][0])
    rows = [{"epis_idx": ep, "step_idx": 250, "exec_start_idx": 0}]
    ds = _make(rows, event_drop=1.0)
    assert not ds[0]["static_mask"].any()


def test_all_task_shapes_are_consistent(cache):
    f = EventMemoryFeatures(cache.n_symbols, cache.max_log)
    keys = list(cache._row)[:: max(1, len(cache._row) // 100)]
    for e, t in keys:
        out = f(*cache[(e, t)])
        assert out["static_image_emb"].shape == (cache.max_log, cache.n_symbols)
        assert int(out["static_mask"].sum()) == min(cache[(e, t)][1], cache.max_log)


def test_dataset_survives_a_pickle_roundtrip(cache):
    """torch spawns data-loader workers and pickles the dataset to reach them.

    A class defined inside a factory function fails here with "Can't pickle local
    object" the moment ``num_workers > 0`` -- which is every real training run, and
    which cost a GPU job that had already loaded 12 GB of weights.
    """
    import functools
    import pickle

    ep = int(cache.meta["train_episodes"][0])
    rows = [{"epis_idx": ep, "step_idx": 250, "exec_start_idx": 0}]
    ds = _make(rows)
    back = pickle.loads(pickle.dumps(ds))
    assert np.array_equal(back[0]["static_image_emb"], ds[0]["static_image_emb"])

    # install() hands a functools.partial to the loader; that has to pickle too.
    bound = functools.partial(_EventFake, event_tag=TAG, event_mode="log")
    assert pickle.loads(pickle.dumps(bound))(rows, None, None, 20)[0]["static_mask"] \
        .shape == (cache.max_log,)
