"""Serve pi0.5 with the event log rebuilt online, without editing the upstream policy.

``MME_VLA_Policy._prepare_history`` fills the four ``static_*`` keys from a visual memory
buffer. This subclass fills them from :class:`RolloutEventLog` instead, driven by the
actions the policy itself commanded. The upstream buffer is still built, but only for the
one thing it does that we need: encoding each frame to ``image_emb_2x2`` with pi0.5's own
SigLIP, which is exactly the quantity the preprocessed dataset stores and therefore
exactly what the tokenizer was fitted on.

Reconstructing the executed action sequence is the part that has to be right. ``infer``
returns a 20-step chunk of which the client executes 16 and then asks again, so the
policy knows every action that was executed and the order they went out in -- but only if
it records the chunk it returned and lines it up with the states that come back through
``add_buffer``. Coding one row per ``infer`` call instead would give a sixteenth of the
codes and a log that is empty at test time and populated at training time.
"""

from __future__ import annotations

import numpy as np

from .features import EventMemoryFeatures
from .online import EventTokenizerBundle, RolloutEventLog
from .task_id import task_of_prompt

VISION_EMB_DIM = 2048
POS_EMB_DIM = 768
STATE_EMB_DIM = 8


class EventMemoryPolicyMixin:
    """Fills pi0.5's memory slots from the log this rollout has produced so far."""

    event_tag: str = "joint_k64"
    event_mode: str = "log"

    # -------------------------------------------------------------- lifecycle
    def _event_setup(self, tag: str = "joint_k64", mode: str = "log",
                     task: str | None = None) -> None:
        if mode not in ("log", "blank"):
            raise ValueError(
                f"event_mode {mode!r} is not available at rollout time. 'wrong' needs "
                "another episode's log, which does not exist in closed loop; train a "
                "'wrong' policy and evaluate it with mode 'log' instead."
            )
        self.event_tag = tag
        self.event_mode = mode
        self.bundle = EventTokenizerBundle(tag)
        self.event_feat = EventMemoryFeatures(self.bundle.n_symbols, self.bundle.max_log)
        self._event_log: RolloutEventLog | None = None
        self._event_task: str | None = None
        self._pending: list[np.ndarray] = []
        self._pushed = 0
        # Pin the task when the eval runs one at a time; prompt matching is a fallback
        # for a whole-benchmark run, and a fallback is where a silent mismatch hides.
        self._pinned_task = task
        print(
            f"[eventtok] rollout memory: tag={tag} mode={mode} "
            f"vocab={self.bundle.n_symbols} budget={self.bundle.max_log} "
            f"chunk={self.bundle.chunk} min_span={self.bundle.min_span}",
            flush=True,
        )

    def _prepare_mem_buffer(self) -> None:
        from mme_vla_suite.shared.mem_buffer import MemoryBuffer

        # Dimensions from the feature, not from the history config: our config declares
        # img input_dim = vocabulary size, which is what the *model's* memory encoder
        # consumes, not what SigLIP produces.
        self.mem_buffer = MemoryBuffer(
            num_views=1,
            img_emb_dim=VISION_EMB_DIM,
            pos_emb_dim=POS_EMB_DIM,
            state_emb_dim=STATE_EMB_DIM,
            compute_token_drop_score=False,
            token_drop_stride=8,
            prepare_buffer=True,
            vision_enc_fn=self._vision_encode,
        )

    def reset(self) -> None:
        super().reset()
        if getattr(self, "bundle", None) is None:
            return
        self._event_log = None
        self._event_task = None
        self._pending = []
        self._pushed = 0

    # ------------------------------------------------------------------ input
    def _ensure_log(self, prompt: str) -> None:
        if self._event_log is not None:
            return
        if self._pinned_task:
            task, score = self._pinned_task, 1.0
        else:
            task, score = task_of_prompt(prompt)
            if score < 0.75:
                print(
                    f"[eventtok] WARNING: best task match is {task} at ratio {score:.2f} "
                    f"for prompt {prompt!r}; the action scale may be the wrong task's. "
                    "Pass --event-task to pin it.",
                    flush=True,
                )
        self._event_task = task
        self._event_log = RolloutEventLog(self.bundle, task)
        note = "" if score == 1.0 else f" (nearest match, ratio {score:.2f})"
        print(f"[eventtok] task = {task}{note}", flush=True)

    def _frame_vision(self, step_idx: int) -> np.ndarray | None:
        """``image_emb_2x2`` for the base view at this execution step, flattened."""
        feats = self.mem_buffer._history_feats
        entry = feats.get(step_idx) if isinstance(feats, dict) else None
        if entry is None:
            return None
        return np.asarray(entry["image_emb_2x2"][0], dtype=np.float32).ravel()

    def _drain(self) -> None:
        """Push every executed step whose state and frame have both arrived."""
        if self._event_log is None or not self._pending:
            return
        # The buffer indexes absolute steps; the tokenizer indexes execution steps, and
        # a video-demo prefix makes the two differ. Getting this backwards would feed the
        # log the demonstration video instead of the robot's own behaviour.
        start = self.exec_start_idx
        while self._pushed < len(self._pending):
            t = self._pushed
            vision = self._frame_vision(start + t)
            state = self._buffer_state(start + t)
            if vision is None or state is None:
                break
            self._event_log.push(self._pending[t], state, vision)
            self._pushed += 1

    def _buffer_state(self, step_idx: int) -> np.ndarray | None:
        feats = self.mem_buffer._history_feats
        entry = feats.get(step_idx) if isinstance(feats, dict) else None
        if entry is None:
            return None
        return np.asarray(entry["state_emb"], dtype=np.float32).ravel()

    def _prepare_history(self, inputs: dict) -> dict:
        self._ensure_log(inputs.get("prompt", ""))
        self._drain()
        if self.event_mode == "blank" or self._event_log is None:
            inputs.update(self.event_feat.blank())
            return inputs
        inputs.update(self.event_feat(*self._event_log.padded()))
        return inputs

    def infer(self, obs: dict) -> dict:
        out = super().infer(obs)
        # Record the chunk that is about to be executed. The client takes the first
        # obs_horizon actions of it; anything it does not execute is overwritten by the
        # next chunk, so the record is trimmed rather than trusted.
        chunk = np.asarray(out["actions"], dtype=np.float32)
        self._pending = self._pending[: self._chunk_start()] + list(chunk)
        return out

    def _chunk_start(self) -> int:
        """How many steps have actually been executed, per the observation buffer."""
        return max(self.step_idx + 1 - self.exec_start_idx, 0)

    # ---------------------------------------------------------------- reporting
    def event_state(self) -> dict:
        if self._event_log is None:
            return {"task": None, "tokens": [], "overflow": 0}
        return {
            "task": self._event_task,
            "tokens": self._event_log.tokens(),
            "overflow": int(self._event_log.overflow().sum()),
            "codes": len(self._event_log._codes),
        }


def event_policy_class():
    """A ``MME_VLA_Policy`` subclass serving event memory. Imports mme_vla_suite."""
    from mme_vla_suite.policies.policy import MME_VLA_Policy

    class EventMemoryPolicy(EventMemoryPolicyMixin, MME_VLA_Policy):
        pass

    return EventMemoryPolicy


def install(tag: str = "joint_k64", mode: str = "log", task: str | None = None):
    """Make ``create_trained_policy`` return the event-memory policy.

    ``policy_config.py`` refers to the class as ``_policy.MME_VLA_Policy``, an attribute
    lookup on the module, so rebinding the attribute is enough -- no edit upstream and no
    copy of create_trained_policy to drift out of date.
    """
    from mme_vla_suite.policies import policy as _policy

    base = _policy.MME_VLA_Policy

    class EventMemoryPolicy(EventMemoryPolicyMixin, base):
        def __init__(self, *args, **kwargs):
            # Before super().__init__, which calls reset() and _prepare_mem_buffer().
            self._event_setup(tag, mode, task)
            super().__init__(*args, **kwargs)

    _policy.MME_VLA_Policy = EventMemoryPolicy
    return EventMemoryPolicy
