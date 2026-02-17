# heimdall/core/decision.py

from heimdall.core.types import REQUEST, SILENT, Label

CONF_THRESHOLD: float = 0.38


def decide(label: Label, confidence: float) -> Label:
    """
    Final confidence gate.

    Rules:
    - Only suppress SILENT predictions.
    - Never downgrade REQUEST / HOSTILE / TOPIC_RESET.
    """
    if label == SILENT and confidence < CONF_THRESHOLD:
        return REQUEST

    return label
