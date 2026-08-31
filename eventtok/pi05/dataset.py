"""Feed the event log into robomme_policy_learning's dataset without editing it.

``RoboMMEDataset`` fills ``static_image_emb`` and friends from per-frame visual features
when it is given a history config, and leaves them ``None`` when it is not. The mixin
here takes the second branch -- the parent is constructed with ``history_config=None`` so
it never touches the feature directory -- and fills the same four keys from the event
cache afterwards.

Everything downstream is untouched: the repack transform already forwards those keys,
``HistAugObservation`` already carries them, and ``embed_prefix`` already prepends them
with the memory mask. Swapping the contents of the memory slots is the whole change,
which is also what makes MovieChat, token-dropping and event memory comparable -- same
encoder, same attention, different memory.

The mixin is deliberately free of any ``mme_vla_suite`` import so it can be tested
without JAX. The concrete subclass lives in :mod:`eventtok.pi05.dataset_impl`, at module
level rather than inside a factory: torch's data loader spawns workers and pickles the
dataset by qualified name, and a class defined inside a function cannot be pickled.
"""

from __future__ import annotations

import numpy as np

from .features import EventMemoryFeatures
from .tokens import EventLogCache


class EventMemoryMixin:
    """Serves the event log through the four ``static_*`` keys.

    Mix in front of a ``RoboMMEDataset``-shaped class::

        class EventRoboMMEDataset(EventMemoryMixin, RoboMMEDataset): ...
    """

    def __init__(
        self,
        dataset_path,
        data_config,
        history_config,
        action_horizon,
        compute_norm_stats: bool = False,
        *,
        event_tag: str = "joint",
        event_drop: float = 0.0,
        event_mode: str = "log",
        event_seed: int = 0,
    ) -> None:
        # history_config=None: the parent must not run its own memory preparation,
        # which would load per-frame SigLIP features we do not use.
        super().__init__(
            dataset_path=dataset_path,
            data_config=data_config,
            history_config=None,
            action_horizon=action_horizon,
            compute_norm_stats=compute_norm_stats,
        )
        self.event_config = history_config
        self.event_tag = event_tag
        self.event_drop = float(event_drop)
        self.event_mode = event_mode
        if event_mode not in ("log", "wrong", "blank"):
            raise ValueError(f"unknown event_mode {event_mode!r}")
        self.event_seed = int(event_seed)
        self._reset_event_state()

    def _reset_event_state(self) -> None:
        self._rng = np.random.default_rng(self.event_seed)
        self._caches: dict[str, EventLogCache] = {}
        self._feat: EventMemoryFeatures | None = None
        self._index = None

    def __getstate__(self):
        # Caches are hundreds of MB of npz and a worker can reopen them lazily; the
        # Generator would also be copied identically into every worker.
        state = dict(self.__dict__)
        for k in ("_rng", "_caches", "_feat", "_index"):
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reset_event_state()

    # ------------------------------------------------------------------ caches
    def _cache_for(self, task: str) -> EventLogCache:
        if task not in self._caches:
            c = EventLogCache(task, self.event_tag)
            if self._feat is None:
                self._feat = EventMemoryFeatures(c.n_symbols, c.max_log)
            elif self._feat.n_symbols != c.n_symbols:
                raise ValueError(
                    f"{task}'s cache has {c.n_symbols} symbols but an earlier task had "
                    f"{self._feat.n_symbols}. One policy over several tasks needs one "
                    "vocabulary: rebuild with build_pi05_joint."
                )
            self._caches[task] = c
        return self._caches[task]

    def _task_of(self, epis_idx: int) -> str:
        from ..data.index import RoboMMEIndex

        if self._index is None:
            self._index = RoboMMEIndex()
        return self._index.task_of(int(epis_idx))

    # ------------------------------------------------------------------ getitem
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        epis_idx = int(np.asarray(data["epis_idx"]).reshape(-1)[0])
        step_idx = int(np.asarray(data["step_idx"]).reshape(-1)[0])
        exec_start = int(np.asarray(data["exec_start_idx"]).reshape(-1)[0])
        # step_idx counts from the start of the recording; the cache is indexed by
        # execution frame. They coincide only when exec_start is 0, which is true of 3
        # of 16 tasks -- this subtraction is why the other 13 are not silently reading
        # the demonstration video's log.
        frame = step_idx - exec_start

        cache = self._cache_for(self._task_of(epis_idx))
        if self.event_mode == "blank" or (
            self.event_drop and self._rng.random() < self.event_drop
        ):
            data.update(self._feat.blank())
            return data

        if self.event_mode == "wrong":
            # The control that decides whether a gain is about the log's content.
            # Another episode of the same task: same vocabulary, same length
            # statistics, wrong history.
            pool = sorted({e for (e, _) in cache._row} - {epis_idx})
            epis_idx = int(pool[self._rng.integers(len(pool))])

        toks, length, ovf = cache.get(epis_idx, frame)
        data.update(self._feat(toks, length, ovf))
        return data


def event_dataset_class():
    """The concrete ``RoboMMEDataset`` subclass. Imports mme_vla_suite."""
    from .dataset_impl import EventRoboMMEDataset

    return EventRoboMMEDataset


def install(module=None, **kwargs):
    """Point ``create_data_loader`` at the event dataset.

    ``dataloader.py`` binds ``RoboMMEDataset`` by ``from ... import``, so rebinding the
    name in that module is what takes effect. Done here rather than by editing upstream:
    robomme_policy_learning is not ours, and a one-line rebind is auditable in a way a
    fork is not.
    """
    if module is None:
        from mme_vla_suite.training import dataloader as module

    cls = event_dataset_class()
    if kwargs:
        import functools

        # functools.partial pickles through to the class, which is module-level, so
        # torch's spawned data-loader workers can rebuild it.
        cls = functools.partial(cls, **kwargs)
    module.RoboMMEDataset = cls
    return cls
