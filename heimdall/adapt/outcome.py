# adapt/outcome.py


from heimdall.core.types import (
    ACK_MINIMAL,
    CONTINUED,
    ESCALATED,
    NONE,
    Outcome,
    SystemAction,
)


def infer_outcome(
    prev_action: SystemAction,
    next_input: str | None,
    time_gap: float,
) -> Outcome:
    if next_input is None:
        return NONE

    # user escalated after minimal acknowledgement
    if len(next_input) > 20 and prev_action == ACK_MINIMAL:
        return ESCALATED

    return CONTINUED
