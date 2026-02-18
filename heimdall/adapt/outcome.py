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
    *,
    confidence_threshold: float = 0.4,
    min_next_len: int = 20,
) -> Outcome:
    if next_input is None:
        return NONE

    # user escalated after minimal acknowledgement
    if confidence < confidence_threshold and len(next_input) > min_next_len:
        return ESCALATED

    return CONTINUED
