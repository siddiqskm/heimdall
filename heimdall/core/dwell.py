# core/dwell.py

from typing import Dict
from core.types import Label
from core.decision import CONF_THRESHOLD

# ---- label constants (single source of truth) ----
SILENT: Label = "SILENT"
STEER: Label = "STEER"
REQUEST: Label = "REQUEST"
TOPIC_RESET: Label = "TOPIC_RESET"
HOSTILE: Label = "HOSTILE"


class LabelDwell:
    """
    State-correct dwell controller (SOFT limits only).

    Invariants:
    - Early low-confidence REQUESTs are suppressed
    - After first intent, REQUESTs are trusted
    - TOPIC_RESET explicitly interrupts intent
    - HOSTILE temporarily suppresses (recoverable)
    - Stability is tracked per label
    """

    HOSTILE_COOLDOWN: int = 2  # number of calm turns to recover

    def __init__(self) -> None:
        self.last_label: Dict[str, Label] = {}
        self.in_intent: Dict[str, bool] = {}
        self.post_reset: Dict[str, bool] = {}
        self.has_seen_intent: Dict[str, bool] = {}

        # ---- hostility (soft) ----
        self.in_hostile: Dict[str, bool] = {}
        self._hostile_cooldown: Dict[str, int] = {}

        # ---- stability tracking ----
        self._stable_turns: Dict[str, int] = {}
        self._stable_label: Dict[str, Label] = {}

    def stable_turns(self, user_id: str, label: Label) -> int:
        if self._stable_label.get(user_id) == label:
            return self._stable_turns.get(user_id, 0)
        return 0

    def _update_stability(self, user_id: str, label: Label) -> None:
        if self._stable_label.get(user_id) == label:
            self._stable_turns[user_id] += 1
        else:
            self._stable_label[user_id] = label
            self._stable_turns[user_id] = 1

    def _recover_from_hostile(self, user_id: str) -> None:
        """
        Reduce hostility cooldown and recover when it reaches zero.
        """
        if not self.in_hostile[user_id]:
            return

        self._hostile_cooldown[user_id] -= 1
        if self._hostile_cooldown[user_id] <= 0:
            self.in_hostile[user_id] = False
            self._hostile_cooldown[user_id] = 0

    def apply(
        self,
        user_id: str,
        predicted: Label,
        activation: float,
    ) -> Label:

        # --- initialize user ---
        if user_id not in self.last_label:
            self.last_label[user_id] = SILENT
            self.in_intent[user_id] = False
            self.post_reset[user_id] = False
            self.has_seen_intent[user_id] = False

            self.in_hostile[user_id] = False
            self._hostile_cooldown[user_id] = 0

            self._stable_turns[user_id] = 0
            self._stable_label[user_id] = SILENT

        # =========================================================
        # HOSTILE ENTRY (soft)
        # =========================================================
        if predicted == HOSTILE:
            self.last_label[user_id] = HOSTILE
            self.in_intent[user_id] = False
            self.post_reset[user_id] = False

            self.in_hostile[user_id] = True
            self._hostile_cooldown[user_id] = self.HOSTILE_COOLDOWN

            self._update_stability(user_id, HOSTILE)
            return HOSTILE

        # =========================================================
        # HOSTILE RECOVERY PATH
        # =========================================================
        if self.in_hostile[user_id]:
            self._recover_from_hostile(user_id)
            self._update_stability(user_id, HOSTILE)
            return HOSTILE

        # --- early low-confidence REQUEST suppression ---
        if (
            predicted == REQUEST
            and activation < CONF_THRESHOLD
            and not self.has_seen_intent[user_id]
        ):
            predicted = SILENT

        # =========================================================
        # TOPIC RESET (always allowed)
        # =========================================================
        if predicted == TOPIC_RESET:
            self.last_label[user_id] = TOPIC_RESET
            self.in_intent[user_id] = False
            self.post_reset[user_id] = True
            self._update_stability(user_id, TOPIC_RESET)
            return TOPIC_RESET

        # --- post-reset mode ---
        if self.post_reset[user_id]:
            if predicted == REQUEST:
                self.post_reset[user_id] = False
                self.in_intent[user_id] = True
                self.has_seen_intent[user_id] = True
                self.last_label[user_id] = REQUEST
                self._update_stability(user_id, REQUEST)
                return REQUEST

            self.last_label[user_id] = SILENT
            self._update_stability(user_id, SILENT)
            return SILENT

        # --- active intent ---
        if self.in_intent[user_id]:
            if predicted == REQUEST:
                self.last_label[user_id] = REQUEST
                self._update_stability(user_id, REQUEST)
                return REQUEST

            if predicted in {SILENT, STEER}:
                self._update_stability(user_id, self.last_label[user_id])
                return self.last_label[user_id]

            self.in_intent[user_id] = False
            self.last_label[user_id] = SILENT
            self._update_stability(user_id, SILENT)
            return SILENT

        # --- idle → intent entry ---
        if predicted == REQUEST:
            self.in_intent[user_id] = True
            self.has_seen_intent[user_id] = True
            self.last_label[user_id] = REQUEST
            self._update_stability(user_id, REQUEST)
            return REQUEST

        self.last_label[user_id] = SILENT
        self._update_stability(user_id, SILENT)
        return SILENT
