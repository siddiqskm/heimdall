# adapt/outcome.py

from typing import Optional

from core.types import (
    SystemAction,
    Outcome,
    NONE,
    ESCALATED,
    CONTINUED,
    ACK_MINIMAL,
)


def infer_outcome(
    prev_action: SystemAction,
    next_input: Optional[str],
    time_gap: float,
) -> Outcome:
    if next_input is None:
        return NONE

    # user escalated after minimal acknowledgement
    if len(next_input) > 20 and prev_action == ACK_MINIMAL:
        return ESCALATED

    return CONTINUED
