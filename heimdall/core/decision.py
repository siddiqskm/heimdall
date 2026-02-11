# core/decision.py

from heimdall.core.types import Label

# ---- label constants (runtime + typed) ----
SILENT: Label = "SILENT"

CONF_THRESHOLD: float = 0.30


def decide(label: Label, confidence: float) -> Label:
    """
    Final confidence gate.

    Rules:
    - Only suppress SILENT predictions.
    - Never downgrade REQUEST / STEER / HOSTILE / TOPIC_RESET.
    """
    if label == SILENT and confidence < CONF_THRESHOLD:
        return SILENT

    return label
