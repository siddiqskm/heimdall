# adapt/learner.py

from typing import Dict

from core.types import Label, Outcome, ESCALATED, NONE
from adapt.config import MAX_BIAS, DECAY, REWARD, PENALTY


def apply_decay(user_delta: Dict[Label, float]) -> None:
    """
    Apply exponential decay to all user bias values.
    """
    for label in user_delta:
        user_delta[label] *= DECAY


def clamp(value: float) -> float:
    """
    Clamp bias values to allowed bounds.
    """
    return max(-MAX_BIAS, min(MAX_BIAS, value))


def update_user_delta(
    user_delta: Dict[Label, float],
    label: Label,
    outcome: Outcome,
) -> None:
    """
    Update user bias based on learning outcome.
    """

    # default zero
    user_delta.setdefault(label, 0.0)

    if outcome == ESCALATED:
        user_delta[label] -= PENALTY
    elif outcome == NONE:
        user_delta[label] += REWARD

    user_delta[label] = clamp(user_delta[label])
