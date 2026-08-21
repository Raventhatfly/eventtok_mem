"""Filesystem locations.

Only the hankyang_lab copy of RoboMME has the per-frame feature files; the
ydu_lab copy's ``features/episode_*/`` directories are empty. Pointing at the
wrong root fails silently (zero features found), so the root is pinned here
rather than passed around.
"""

from __future__ import annotations

import os
from pathlib import Path

# The copy that actually has features/. Override with EVENTTOK_ROBOMME_ROOT.
ROBOMME_ROOT = Path(
    os.environ.get(
        "EVENTTOK_ROBOMME_ROOT",
        "/n/netscratch/hankyang_lab/Lab/felix/dataset/robomme",
    )
)

PREPROC = ROBOMME_ROOT / "robomme_preprocessed_data"
PKL_DIR = PREPROC / "data"
FEATURE_DIR = PREPROC / "features"
INDEX_JSON = ROBOMME_ROOT / "robomme_data_h5" / "fluxvla_hdf5_index.json"

# Repacked per-episode features and other build artifacts. Never on /n/home04
# (81% full); default to lab scratch.
CACHE_ROOT = Path(
    os.environ.get(
        "EVENTTOK_CACHE_ROOT",
        "/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache",
    )
)


def check_root() -> None:
    """Fail loudly if the data root is the copy without features."""
    if not INDEX_JSON.is_file():
        raise FileNotFoundError(f"no index json at {INDEX_JSON}")
    probe = FEATURE_DIR / "episode_0"
    if not probe.is_dir() or not any(probe.glob("token_emb_*.npy")):
        raise FileNotFoundError(
            f"{probe} has no token_emb_*.npy files. This is almost certainly "
            f"the ydu_lab copy of RoboMME, whose feature dirs are empty. Set "
            f"EVENTTOK_ROBOMME_ROOT to the hankyang_lab copy."
        )
