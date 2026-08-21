"""Recover the target repetition count N from a task prompt.

RoboMME stores no numeric count field, so N lives only in the instruction text.
Both counting tasks phrase it differently, and in both a *missing* phrase means
N = 1 rather than N = 0.

Verified distributions over the 100 episodes of each task:
    PickXtimes   N in {1,2,3,4,5}  ->  23 / 25 / 27 / 12 / 13
    SwingXtimes  N in {1,2,3}      ->  23 / 31 / 46

SwingXtimes topping out at N = 3 is why the extrapolation experiment has to run
on PickXtimes: training on {1,2,3} and holding out {4,5} is only possible there.
"""

from __future__ import annotations

import re

_WORD_TO_INT = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

# One pattern per task that has a repetition count. Tasks absent from this map
# have no count semantics at all.
_COUNT_PATTERNS = {
    "PickXtimes": re.compile(r"repeating this action (\w+) times"),
    "SwingXtimes": re.compile(r"repeating this back and forth motion (\w+) times"),
}

_COLOR = re.compile(r"the (red|green|blue) cube")


def has_count(task: str) -> bool:
    return task in _COUNT_PATTERNS


def extract_count(task: str, prompt: str) -> int | None:
    """Target repetition count, or None if the task has no count semantics.

    A prompt with no matching phrase means the action is performed once, which
    is a real N = 1 episode and not a parse failure.
    """
    pattern = _COUNT_PATTERNS.get(task)
    if pattern is None:
        return None
    match = pattern.search(prompt)
    if match is None:
        return 1
    word = match.group(1).lower()
    if word not in _WORD_TO_INT:
        raise ValueError(
            f"unparsed count word {word!r} in {task} prompt: {prompt!r}. "
            f"Extend _WORD_TO_INT rather than silently defaulting."
        )
    return _WORD_TO_INT[word]


def extract_color(prompt: str) -> str | None:
    match = _COLOR.search(prompt)
    return match.group(1) if match else None
