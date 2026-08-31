"""Serve a pi0.5 checkpoint whose memory is the event log, rebuilt as it acts.

    python -m eventtok.scripts.serve_pi05_event --event-tag=joint_k64 \
        --port=8000 policy:checkpoint --policy.dir=<ckpt>/79999 \
        --policy.config=mme_vla_suite

Same arguments as robomme_policy_learning's scripts/serve_policy.py, plus the event
flags. It rebinds ``policy.MME_VLA_Policy`` and then calls the upstream main, so the
checkpoint loading, transforms and websocket server are all theirs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROBOMME = Path(os.environ.get("ROBOMME_REPO", "/n/home04/wfy/repos/robomme_policy_learning"))


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--event-tag", default="joint_k64")
    ap.add_argument("--event-mode", default="log", choices=["log", "blank"])
    ap.add_argument("--event-task", default=None,
                    help="pin the task instead of inferring it from the prompt")
    ours, rest = ap.parse_known_args()

    if not (ROBOMME / "scripts" / "serve_policy.py").is_file():
        raise SystemExit(f"robomme_policy_learning not at {ROBOMME}; set ROBOMME_REPO")
    sys.path.insert(0, str(ROBOMME))
    os.chdir(ROBOMME)

    from ..pi05.policy import install

    install(tag=ours.event_tag, mode=ours.event_mode, task=ours.event_task)

    import tyro

    import scripts.serve_policy as serve

    sys.argv = [sys.argv[0], *rest]
    serve.main(tyro.cli(serve.Args))


if __name__ == "__main__":
    main()
