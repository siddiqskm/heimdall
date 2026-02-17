# heimdall/adapt/learner.py


from heimdall.adapt.config import DECAY, MAX_BIAS, PENALTY, REWARD
from heimdall.core.types import ESCALATED, NONE, Label, Outcome


def apply_decay(user_delta: dict[Label, float]) -> None:
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
    user_delta: dict[Label, float],
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
