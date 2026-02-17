# heimdall/core/dwell.py

from enum import StrEnum

from heimdall.core.decision import CONF_THRESHOLD
from heimdall.core.types import HOSTILE, REQUEST, SILENT, TOPIC_RESET, Label


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

    HOSTILE_COOLDOWN: int = 2
    INTENT_DECAY_SILENT_STREAK: int = 2

    def __init__(self, debug: bool = False) -> None:

        self.state: dict[str, DwellState] = {}
        self._hostile_cooldown: dict[str, int] = {}
        self._hostile_streak: dict[str, int] = {}

        self._predicted_silent_streak: dict[str, int] = {}

        self.debug = debug

    # =========================================================

    def _init_user(self, user_id: str) -> None:
        if user_id in self.state:
            return

        self.state[user_id] = DwellState.IDLE
        self._hostile_cooldown[user_id] = 0
        self._hostile_streak[user_id] = 0
        self._predicted_silent_streak[user_id] = 0

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

        print(
            f"[DWELL DEBUG] user={user_id} "
            f"prev={prev_state} pred={predicted} "
            f"act={activation:.2f} "
            f"cooldown={self._hostile_cooldown[user_id]} "
            f"hstreak={self._hostile_streak[user_id]} "
            f"sstreak={self._predicted_silent_streak[user_id]} "
            f"next={self.state[user_id]} final={final}"
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
            self._hostile_cooldown[user_id] = self.HOSTILE_COOLDOWN
            self._predicted_silent_streak[user_id] = 0

            final = HOSTILE

        # =====================================================
        # HOSTILE STATE
        # =====================================================

        elif prev_state == DwellState.HOSTILE:

            if predicted == HOSTILE:
                self._hostile_streak[user_id] += 1
                final = HOSTILE

            else:
                # streak ended
                streak = self._hostile_streak[user_id]

                if streak >= 2:
                    # immediate recovery allowed
                    self._hostile_streak[user_id] = 0
                    self._hostile_cooldown[user_id] = 0

                    if predicted == REQUEST and activation >= CONF_THRESHOLD:
                        self.state[user_id] = DwellState.INTENT
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

                        if predicted == REQUEST and activation >= CONF_THRESHOLD:
                            self.state[user_id] = DwellState.INTENT
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

            if predicted == REQUEST and activation >= CONF_THRESHOLD:
                self.state[user_id] = DwellState.INTENT
                final = REQUEST
            else:
                final = SILENT

        # =====================================================
        # INTENT STATE
        # =====================================================

        elif prev_state == DwellState.INTENT:

            if predicted == REQUEST:
                self._predicted_silent_streak[user_id] = 0
                final = REQUEST

            elif predicted == SILENT:

                # high-activation SILENT = acknowledgement
                if activation >= CONF_THRESHOLD:
                    self._predicted_silent_streak[user_id] = 0
                    final = REQUEST

                else:
                    self._predicted_silent_streak[user_id] += 1

                    if (
                        self._predicted_silent_streak[user_id]
                        >= self.INTENT_DECAY_SILENT_STREAK
                    ):
                        self.state[user_id] = DwellState.IDLE
                        final = SILENT
                    else:
                        final = REQUEST

            elif predicted == HOSTILE:
                self.state[user_id] = DwellState.HOSTILE
                self._hostile_streak[user_id] = 1
                self._hostile_cooldown[user_id] = self.HOSTILE_COOLDOWN
                final = HOSTILE

            elif predicted == TOPIC_RESET:
                self.state[user_id] = DwellState.POST_RESET
                final = TOPIC_RESET

        # =====================================================
        # IDLE STATE
        # =====================================================

        elif prev_state == DwellState.IDLE:

            if predicted == REQUEST and activation >= CONF_THRESHOLD:
                self.state[user_id] = DwellState.INTENT
                final = REQUEST
            else:
                final = SILENT

        # =====================================================

        self._debug(user_id, prev_state, predicted, activation, final)
        return final
