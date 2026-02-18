# heimdall/core/dwell.py

import logging
from enum import StrEnum

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


class LabelDwell:
    """
    Deterministic FSM dwell controller.

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

        self.state: dict[str, DwellState] = {}
        self._hostile_cooldown: dict[str, int] = {}
        self._hostile_streak: dict[str, int] = {}

        self._predicted_silent_streak: dict[str, int] = {}
        self._high_silent_in_intent: dict[str, int] = {}
        self._intent_from_hostile_recovery: dict[str, bool] = {}

        self.debug = debug

    # =========================================================

    def _init_user(self, user_id: str) -> None:
        if user_id in self.state:
            return

        self.state[user_id] = DwellState.IDLE
        self._hostile_cooldown[user_id] = 0
        self._hostile_streak[user_id] = 0
        self._predicted_silent_streak[user_id] = 0
        self._high_silent_in_intent[user_id] = 0
        self._intent_from_hostile_recovery[user_id] = False

    # =========================================================

    def _debug(
        self,
        user_id: str,
        prev_state: DwellState,
        predicted: Label,
        activation: float,
        final: Label,
    ) -> None:
        if not self.debug:
            return

        logger.debug(
            "user=%s prev=%s pred=%s act=%.2f cooldown=%s hstreak=%s sstreak=%s next=%s final=%s",
            user_id,
            prev_state,
            predicted,
            activation,
            self._hostile_cooldown[user_id],
            self._hostile_streak[user_id],
            self._predicted_silent_streak[user_id],
            self.state[user_id],
            final,
        )

    # =========================================================

    def apply(
        self,
        user_id: str,
        predicted: Label,
        activation: float,
    ) -> Label:

        self._init_user(user_id)

        prev_state = self.state[user_id]
        final: Label = SILENT

        # =====================================================
        # HOSTILE ENTRY
        # =====================================================

        if predicted == HOSTILE:

            self.state[user_id] = DwellState.HOSTILE
            self._hostile_streak[user_id] += 1
            self._hostile_cooldown[user_id] = self._hostile_cooldown_turns
            self._predicted_silent_streak[user_id] = 0

            final = HOSTILE

        # =====================================================
        # HOSTILE STATE
        # =====================================================

        elif prev_state == DwellState.HOSTILE:

            if predicted == HOSTILE:
                self._hostile_streak[user_id] += 1
                final = HOSTILE

            elif predicted == REQUEST and activation < self._hostile_recovery_threshold:
                # Borderline REQUEST (e.g. "im done") in hostile context → keep HOSTILE
                final = HOSTILE

            else:
                # streak ended, clear REQUEST or SILENT
                streak = self._hostile_streak[user_id]

                if streak >= 2:
                    # immediate recovery allowed only with clear REQUEST signal
                    self._hostile_streak[user_id] = 0
                    self._hostile_cooldown[user_id] = 0

                    if (
                        predicted == REQUEST
                        and activation >= self._hostile_recovery_threshold
                    ):
                        self.state[user_id] = DwellState.INTENT
                        self._high_silent_in_intent[user_id] = 0
                        self._intent_from_hostile_recovery[user_id] = True
                        final = REQUEST
                    else:
                        self.state[user_id] = DwellState.IDLE
                        final = SILENT

                else:
                    # single hostile → cooldown required
                    cooldown = self._hostile_cooldown[user_id]

                    if cooldown > 0:
                        self._hostile_cooldown[user_id] -= 1
                        final = HOSTILE
                    else:
                        self._hostile_streak[user_id] = 0

                        if (
                            predicted == REQUEST
                            and activation >= self._hostile_recovery_threshold
                        ):
                            self.state[user_id] = DwellState.INTENT
                            self._high_silent_in_intent[user_id] = 0
                            self._intent_from_hostile_recovery[user_id] = True
                            final = REQUEST
                        else:
                            self.state[user_id] = DwellState.IDLE
                            final = SILENT

        # =====================================================
        # TOPIC RESET
        # =====================================================

        elif predicted == TOPIC_RESET:

            self.state[user_id] = DwellState.POST_RESET
            self._hostile_streak[user_id] = 0
            self._hostile_cooldown[user_id] = 0
            self._predicted_silent_streak[user_id] = 0

            final = TOPIC_RESET

        # =====================================================
        # POST RESET
        # =====================================================

        elif prev_state == DwellState.POST_RESET:

            if predicted == REQUEST and activation >= self._confidence_threshold:
                self.state[user_id] = DwellState.INTENT
                self._high_silent_in_intent[user_id] = 0
                self._intent_from_hostile_recovery[user_id] = False
                final = REQUEST
            else:
                final = SILENT

        # =====================================================
        # INTENT STATE
        # =====================================================

        elif prev_state == DwellState.INTENT:

            if predicted == REQUEST:
                self._predicted_silent_streak[user_id] = 0
                self._high_silent_in_intent[user_id] = 0
                self._intent_from_hostile_recovery[user_id] = False
                final = REQUEST

            elif predicted == SILENT:

                # high-activation SILENT = acknowledgement (or decay if from hostile recovery)
                if activation >= self._confidence_threshold:
                    self._predicted_silent_streak[user_id] = 0
                    self._high_silent_in_intent[user_id] += 1
                    if (
                        self._high_silent_in_intent[user_id] >= 2
                        and self._intent_from_hostile_recovery[user_id]
                    ):
                        self.state[user_id] = DwellState.IDLE
                        self._high_silent_in_intent[user_id] = 0
                        self._intent_from_hostile_recovery[user_id] = False
                        final = SILENT
                    else:
                        final = REQUEST

                else:
                    self._high_silent_in_intent[user_id] = 0
                    self._predicted_silent_streak[user_id] += 1

                    if (
                        self._predicted_silent_streak[user_id]
                        >= self._intent_decay_silent_streak
                    ):
                        self.state[user_id] = DwellState.IDLE
                        final = SILENT
                    else:
                        final = REQUEST

            elif predicted == HOSTILE:
                self.state[user_id] = DwellState.HOSTILE
                self._hostile_streak[user_id] = 1
                self._hostile_cooldown[user_id] = self._hostile_cooldown_turns
                final = HOSTILE

            elif predicted == TOPIC_RESET:
                self.state[user_id] = DwellState.POST_RESET
                final = TOPIC_RESET

        # =====================================================
        # IDLE STATE
        # =====================================================

        elif prev_state == DwellState.IDLE:

            if predicted == REQUEST and activation >= self._confidence_threshold:
                self.state[user_id] = DwellState.INTENT
                final = REQUEST
            else:
                final = SILENT

        # =====================================================

        self._debug(user_id, prev_state, predicted, activation, final)
        return final
