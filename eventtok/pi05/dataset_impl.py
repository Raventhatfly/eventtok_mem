"""The concrete event dataset. Separate module so importing the mixin needs no JAX.

Module level, not built by a factory: torch's data loader spawns workers and pickles the
dataset by qualified name. A class defined inside a function fails with
``Can't pickle local object`` the moment ``num_workers > 0``.
"""

from __future__ import annotations

from mme_vla_suite.training.dataset import RoboMMEDataset

from .dataset import EventMemoryMixin


class EventRoboMMEDataset(EventMemoryMixin, RoboMMEDataset):
    pass
