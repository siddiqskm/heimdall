# heimdall/core/dwell.py

import json
import logging
import uuid
from enum import StrEnum
from pathlib import Path

from heimdall.core.config import HeimdallConfig
from heimdall.core.types import HOSTILE, REQUEST, SILENT, TOPIC_RESET, Label

logger = logging.getLogger(__name__)


class DwellState(StrEnum):
    """
    Explicit FSM states for dwell controller.

    IDLE       → no active intent
    INTENT     → active topic intent
    HOSTILE    → hostile suppression active
    POST_RESET → explicit reset boundary
    """

    IDLE = "IDLE"
    INTENT = "INTENT"
    HOSTILE = "HOSTILE"
    POST_RESET = "POST_RESET"


def _load_dwell_state(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _save_dwell_state(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


class LabelDwell:
    """
    Deterministic FSM dwell controller. One instance per chat.

    Invariants enforced:
    - Early low-confidence REQUESTs suppressed
    - Intent continuity preserved
    - High-activation SILENT inside INTENT treated as acknowledgement
    - Intent decays only on low-activation silent streak
    - Hostile streak ≥2 allows immediate recovery
    - Hostile never permanently locks
    - Reset always wins
    """

    def __init__(
        self,
        config: HeimdallConfig | None = None,
        chat_id: str | None = None,
        debug: bool = False,
    ) -> None:
        c = config
        self._confidence_threshold: float = (
            c.confidence_threshold if c else 0.38
        )
        self._hostile_cooldown_turns: int = (
            c.hostile_cooldown if c else 2
        )
        self._intent_decay_silent_streak: int = (
            c.intent_decay_silent_streak if c else 2
        )
        self._hostile_recovery_threshold: float = (
            c.hostile_recovery_threshold if c else 0.5
        )

        self._chat_id: str = chat_id if chat_id else uuid.uuid4().hex
        self._dwell_path: Path | None = None
        if c is not None:
            c.ensure_chat_dir(self._chat_id)
            self._dwell_path = c.chat_dir(self._chat_id) / "dwell.json"

        # Single chat state (no dicts)
        self._state = DwellState.IDLE
        self._hostile_cooldown = 0
        self._hostile_streak = 0
        self._predicted_silent_streak = 0
        self._high_silent_in_intent = 0
        self._intent_from_hostile_recovery = False

        # Consecutive same-label count for stable_turns (not persisted)
        self._consecutive_label: Label | None = None
        self._consecutive_count = 0

        loaded = _load_dwell_state(self._dwell_path) if self._dwell_path else None
        if loaded:
            self._state = DwellState(loaded.get("state", self._state.value))
            self._hostile_cooldown = int(loaded.get("hostile_cooldown", 0))
            self._hostile_streak = int(loaded.get("hostile_streak", 0))
            self._predicted_silent_streak = int(loaded.get("predicted_silent_streak", 0))
            self._high_silent_in_intent = int(loaded.get("high_silent_in_intent", 0))
            self._intent_from_hostile_recovery = bool(
                loaded.get("intent_from_hostile_recovery", False)
            )

        self.debug = debug

    @property
    def chat_id(self) -> str:
        return self._chat_id

    @property
    def state(self) -> DwellState:
        return self._state

    def persist(self) -> None:
        """Write dwell state to this chat's directory."""
        if not self._dwell_path:
            return
        _save_dwell_state(
            self._dwell_path,
            {
                "state": self._state.value,
                "hostile_cooldown": self._hostile_cooldown,
                "hostile_streak": self._hostile_streak,
                "predicted_silent_streak": self._predicted_silent_streak,
                "high_silent_in_intent": self._high_silent_in_intent,
                "intent_from_hostile_recovery": self._intent_from_hostile_recovery,
            },
        )

    def stable_turns(self, label: Label) -> int:
        """Number of consecutive turns the given label has been output for this chat."""
        if self._consecutive_label == label:
            return self._consecutive_count
        return 0

    def _debug(
        self,
        prev_state: DwellState,
        predicted: Label,
        activation: float,
        final: Label,
    ) -> None:
        if not self.debug:
            return
        logger.debug(
            "chat=%s prev=%s pred=%s act=%.2f cooldown=%s hstreak=%s sstreak=%s next=%s final=%s",
            self._chat_id,
            prev_state,
            predicted,
            activation,
            self._hostile_cooldown,
            self._hostile_streak,
            self._predicted_silent_streak,
            self._state,
            final,
        )

    def apply(
        self,
        predicted: Label,
        activation: float,
    ) -> Label:

        prev_state = self._state
        final: Label = SILENT

        # =====================================================
        # HOSTILE ENTRY
        # =====================================================

        if predicted == HOSTILE:

            self._state = DwellState.HOSTILE
            self._hostile_streak += 1
            self._hostile_cooldown = self._hostile_cooldown_turns
            self._predicted_silent_streak = 0

            final = HOSTILE

        # =====================================================
        # HOSTILE STATE
        # =====================================================

        elif prev_state == DwellState.HOSTILE:

            if predicted == HOSTILE:
                self._hostile_streak += 1
                final = HOSTILE

            elif predicted == REQUEST and activation < self._hostile_recovery_threshold:
                final = HOSTILE

            else:
                streak = self._hostile_streak

                if streak >= 2:
                    self._hostile_streak = 0
                    self._hostile_cooldown = 0

                    if (
                        predicted == REQUEST
                        and activation >= self._hostile_recovery_threshold
                    ):
                        self._state = DwellState.INTENT
                        self._high_silent_in_intent = 0
                        self._intent_from_hostile_recovery = True
                        final = REQUEST
                    else:
                        self._state = DwellState.IDLE
                        final = SILENT

                else:
                    if self._hostile_cooldown > 0:
                        self._hostile_cooldown -= 1
                        final = HOSTILE
                    else:
                        self._hostile_streak = 0

                        if (
                            predicted == REQUEST
                            and activation >= self._hostile_recovery_threshold
                        ):
                            self._state = DwellState.INTENT
                            self._high_silent_in_intent = 0
                            self._intent_from_hostile_recovery = True
                            final = REQUEST
                        else:
                            self._state = DwellState.IDLE
                            final = SILENT

        # =====================================================
        # TOPIC RESET
        # =====================================================

        elif predicted == TOPIC_RESET:

            self._state = DwellState.POST_RESET
            self._hostile_streak = 0
            self._hostile_cooldown = 0
            self._predicted_silent_streak = 0

            final = TOPIC_RESET

        # =====================================================
        # POST RESET
        # =====================================================

        elif prev_state == DwellState.POST_RESET:

            if predicted == REQUEST and activation >= self._confidence_threshold:
                self._state = DwellState.INTENT
                self._high_silent_in_intent = 0
                self._intent_from_hostile_recovery = False
                final = REQUEST
            else:
                final = SILENT

        # =====================================================
        # INTENT STATE
        # =====================================================

        elif prev_state == DwellState.INTENT:

            if predicted == REQUEST:
                self._predicted_silent_streak = 0
                self._high_silent_in_intent = 0
                self._intent_from_hostile_recovery = False
                final = REQUEST

            elif predicted == SILENT:

                if activation >= self._confidence_threshold:
                    self._predicted_silent_streak = 0
                    self._high_silent_in_intent += 1
                    if (
                        self._high_silent_in_intent >= 2
                        and self._intent_from_hostile_recovery
                    ):
                        self._state = DwellState.IDLE
                        self._high_silent_in_intent = 0
                        self._intent_from_hostile_recovery = False
                        final = SILENT
                    else:
                        final = REQUEST

                else:
                    self._high_silent_in_intent = 0
                    self._predicted_silent_streak += 1

                    if (
                        self._predicted_silent_streak
                        >= self._intent_decay_silent_streak
                    ):
                        self._state = DwellState.IDLE
                        final = SILENT
                    else:
                        final = REQUEST

            elif predicted == HOSTILE:
                self._state = DwellState.HOSTILE
                self._hostile_streak = 1
                self._hostile_cooldown = self._hostile_cooldown_turns
                final = HOSTILE

            elif predicted == TOPIC_RESET:
                self._state = DwellState.POST_RESET
                final = TOPIC_RESET

        # =====================================================
        # IDLE STATE
        # =====================================================

        elif prev_state == DwellState.IDLE:

            if predicted == REQUEST and activation >= self._confidence_threshold:
                self._state = DwellState.INTENT
                final = REQUEST
            else:
                final = SILENT

        # =====================================================
        # Consecutive label count for stable_turns
        # =====================================================

        if final == self._consecutive_label:
            self._consecutive_count += 1
        else:
            self._consecutive_label = final
            self._consecutive_count = 1

        self._debug(prev_state, predicted, activation, final)
        self.persist()
        return final
