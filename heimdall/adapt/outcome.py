# heimdall/adapt/outcome.py


from heimdall.core.types import (
    CONTINUED,
    ESCALATED,
    NONE,
    Outcome,
)


def infer_outcome(
    confidence: float,
    next_input: str | None,
) -> Outcome:
    if next_input is None:
        return NONE

    # user escalated after minimal acknowledgement
    if confidence < 0.4 and len(next_input) > 20:
        return ESCALATED

    return CONTINUED
