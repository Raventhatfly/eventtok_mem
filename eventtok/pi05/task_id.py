"""Which RoboMME task is this rollout? The policy server is only told the prompt.

``action_scale`` is per task -- delta actions are divided by it before anything else --
so the online tokenizer cannot code a step without knowing the task. The client knows
(``env_runner.env_id``) but does not send it, and the element dict it does send carries
only images, state and the prompt.

Matching on the prompt works because the environment generates ``info["task_goal"]`` from
the same templates that produced the dataset prompts. It is not exact, though: a fresh
seed can name a different colour or spell a different count, so this scores similarity
against every task's prompt set rather than requiring equality, and says out loud how
confident the match was.
"""

from __future__ import annotations

import difflib
from functools import lru_cache

from ..data.index import RoboMMEIndex


@lru_cache(maxsize=1)
def _prompts_by_task() -> dict[str, list[str]]:
    index = RoboMMEIndex()
    out: dict[str, set[str]] = {}
    for ep in index.episodes:
        if ep.prompt:
            out.setdefault(ep.task, set()).add(ep.prompt.strip())
    return {k: sorted(v) for k, v in out.items()}


def task_of_prompt(prompt: str) -> tuple[str, float]:
    """``(task, similarity)``. Similarity 1.0 means the prompt is in the dataset."""
    p = (prompt or "").strip()
    best, score = None, -1.0
    for task, prompts in _prompts_by_task().items():
        if p in prompts:
            return task, 1.0
        m = difflib.get_close_matches(p, prompts, n=1, cutoff=0.0)
        if m:
            r = difflib.SequenceMatcher(None, p, m[0]).ratio()
            if r > score:
                best, score = task, r
    if best is None:
        raise ValueError(f"no task matched prompt {prompt!r}")
    return best, score
