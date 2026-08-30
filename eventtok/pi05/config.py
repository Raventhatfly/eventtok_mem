"""Generate the history config pi0.5 reads for event memory.

``get_history_config`` does ``os.path.join("src/mme_vla_suite/models/config/robomme",
history_config)``, and ``os.path.join`` returns its second argument unchanged when that
argument is absolute. So an absolute path to a file in *this* repo loads without any
change to robomme_policy_learning.

Fields the upstream code actually reads for our path:

``budget``                 slot count; asserted against ``static_image_emb.shape[1]``
``memory_feature.*.input_dim``  widths of the three blocks the encoder concatenates
``memory_token_dim``       output width, 2048 for the context integration
``use_pos_emb/use_state_emb``   whether pos and state blocks are built at all
``integration_type``       ``context`` prepends the memory to the prefix
``representation_type``    ``perceptual`` selects the generic encoder we ride
``streaming_obs_horizon``  train.py raises unless this is 16
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .features import EventMemoryFeatures
from .tokens import EventLogCache

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "configs" / "pi05"


def history_config(
    n_symbols: int,
    max_log: int,
    *,
    pos_dim: int = 64,
    integration_type: str = "context",
    memory_token_dim: int = 2048,
    pos_hidden: int = 128,
    state_hidden: int = 256,
) -> dict:
    feat = EventMemoryFeatures(n_symbols, max_log, pos_dim)
    dims = feat.dims
    return {
        "budget": int(max_log),
        "num_views": 1,
        "token_per_image": 4,
        "streaming_obs_horizon": 16,
        "pool_type": "mean",
        "use_pos_emb": True,
        # Not optional. The prefix is bidirectional, so without a slot position the log
        # is a set and three swings look like one.
        "use_state_emb": True,
        # Carries the eviction tally, which is what keeps the count exact once the
        # window is full.
        "memory_feature": {
            "img": {"net": "identity", "input_dim": dims["img"]},
            "pos": {"input_dim": dims["pos"], "hidden_dim": pos_hidden},
            "state": {"input_dim": dims["state"], "hidden_dim": state_hidden},
        },
        "integration_type": integration_type,
        "memory_token_dim": memory_token_dim,
        "representation_type": "perceptual",
        # Read only by the upstream dataset, which the event dataset replaces. Kept so
        # the file still loads if someone points the stock pipeline at it.
        "perceptual_memory": {"type": "token_dropping"},
        # Provenance, ignored by the model.
        "event_memory": {"n_symbols": int(n_symbols), "max_log": int(max_log),
                         "pos_dim": int(pos_dim)},
    }


def write_for_cache(task: str, tag: str = "joint", out: Path | None = None,
                    **kwargs) -> Path:
    """Emit the YAML whose dims match a built cache. Run after building the cache."""
    cache = EventLogCache(task, tag)
    cfg = history_config(cache.n_symbols, cache.max_log, **kwargs)
    out = out or DEFAULT_DIR / f"eventtok-{tag}-{cfg['integration_type']}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out
