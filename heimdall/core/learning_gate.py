# heimdall/core/learning_gate.py

import time

from heimdall.core.config import HeimdallConfig
from heimdall.core.types import (
    ALLOW_PROGRESS,
    ESCALATED,
    REQUEST,
    TOPIC_RESET,
    Label,
    Outcome,
    SystemAction,
)


class LearningGate:
    """
    Hard gate controlling whether learning side-effects are allowed.
    """

    def __init__(
        self,
        config: HeimdallConfig | None = None,
        *,
        min_confidence: float = 0.35,
        min_stable_turns: int = 2,
        min_interval_sec: float = 30.0,
    ) -> None:
        if config is not None:
            min_confidence = config.learning_gate_min_confidence
            min_stable_turns = config.learning_gate_min_stable_turns
            min_interval_sec = config.learning_gate_min_interval_sec

        self.min_confidence = min_confidence
        self.min_stable_turns = min_stable_turns
        self.min_interval_sec = min_interval_sec

        # (user_id, label) -> last learn timestamp
        self._last_learn_ts: dict[tuple[str, Label], float] = {}

    def allow(
        self,
        *,
        user_id: str,
        final_label: Label,
        confidence: float,
        stable_turns: int,
        action: SystemAction,
        outcome: Outcome | None,
        now: float | None = None,
    ) -> bool:

        now = now or time.time()

        # ---- Rule 1: label eligibility ----
        if final_label not in {REQUEST, TOPIC_RESET}:
            return False

        # ---- Rule 2: confidence floor ----
        if confidence < self.min_confidence:
            return False

        # ---- Rule 3: stability requirement ----
        if stable_turns < self.min_stable_turns:
            return False

        # ---- Rule 4: outcome validation ----
        if outcome == ESCALATED:
            return False

        # ---- Rule 5: action gate ----
        if action != ALLOW_PROGRESS:
            return False

        # ---- Rule 6: rate limiting ----
        key = (user_id, final_label)
        last_ts = self._last_learn_ts.get(key)

        if last_ts is not None and (now - last_ts) < self.min_interval_sec:
            return False

        self._last_learn_ts[key] = now
        return True
