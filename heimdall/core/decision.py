# heimdall/core/decision.py

from heimdall.core.types import REQUEST, SILENT, Label

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.38


def decide(
    label: Label,
    confidence: float,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Label:
    """
    Final confidence gate.

    Rules:
    - Only suppress SILENT predictions.
    - Never downgrade REQUEST / HOSTILE / TOPIC_RESET.
    """
    if label == SILENT and confidence < confidence_threshold:
        return REQUEST

    return label
