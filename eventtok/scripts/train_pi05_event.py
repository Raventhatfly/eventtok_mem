"""Train pi0.5 with the event log in its memory slot.

    python -m eventtok.scripts.train_pi05_event mme_vla_suite \
        --model.history_config=$PWD/configs/pi05/eventtok-joint-context.yaml \
        --exp-name=eventtok_joint --event-tag=joint

Nothing in robomme_policy_learning is edited. Two seams do the work:

* ``get_history_config`` builds its path with ``os.path.join(<their dir>, arg)``, which
  returns ``arg`` unchanged when it is absolute -- so a config file in this repo loads.
* ``dataloader.py`` binds ``RoboMMEDataset`` with ``from ... import``, so rebinding that
  one name serves event memory instead of visual features. The rest of the pipeline --
  repack transform, ``HistAugObservation``, ``embed_prefix`` -- already carries the
  ``static_*`` keys and needs no change.

``--event-mode`` selects the condition. ``wrong`` is not optional decoration: if a log
from a different episode trains to the same success rate, the policy is not reading the
memory and the ``log`` number means nothing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROBOMME = Path(os.environ.get("ROBOMME_REPO", "/n/home04/wfy/repos/robomme_policy_learning"))


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--event-tag", default="joint")
    ap.add_argument("--event-mode", default="log", choices=["log", "wrong", "blank"])
    ap.add_argument("--event-drop", type=float, default=0.0)
    ap.add_argument("--event-seed", type=int, default=0)
    ours, rest = ap.parse_known_args()

    if not (ROBOMME / "scripts" / "train.py").is_file():
        raise SystemExit(f"robomme_policy_learning not at {ROBOMME}; set ROBOMME_REPO")
    sys.path.insert(0, str(ROBOMME))
    # get_history_config and the checkpoint layout both resolve relative to cwd.
    os.chdir(ROBOMME)

    from mme_vla_suite.training import config as _config
    from mme_vla_suite.training import dataloader as _dl

    from ..pi05.dataset import install

    install(
        _dl,
        event_tag=ours.event_tag,
        event_mode=ours.event_mode,
        event_drop=ours.event_drop,
        event_seed=ours.event_seed,
    )
    print(
        f"[eventtok] memory = {ours.event_mode} (tag={ours.event_tag}, "
        f"drop={ours.event_drop}); dataset = {_dl.RoboMMEDataset}",
        flush=True,
    )

    sys.argv = [sys.argv[0], *rest]
    cfg = _config.cli()
    if cfg.model.history_config is None:
        raise SystemExit(
            "--model.history_config is required; generate it with "
            "python -m eventtok.scripts.write_pi05_config"
        )

    import scripts.train as train

    train.main(cfg)


if __name__ == "__main__":
    main()
