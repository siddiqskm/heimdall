# core/types.py

from typing import Literal, Final

# =====================
# Labels
# =====================

Label = Literal[
    "SILENT",
    "STEER",
    "REQUEST",
    "TOPIC_RESET",
    "HOSTILE",
]

SILENT: Final[Label] = "SILENT"
STEER: Final[Label] = "STEER"
REQUEST: Final[Label] = "REQUEST"
TOPIC_RESET: Final[Label] = "TOPIC_RESET"
HOSTILE: Final[Label] = "HOSTILE"

LABELS: tuple[Label, ...] = (
    SILENT,
    STEER,
    REQUEST,
    TOPIC_RESET,
    HOSTILE,
)

LABEL_TO_ID: dict[Label, int] = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL: dict[int, Label] = {i: label for label, i in LABEL_TO_ID.items()}


# =====================
# System actions
# =====================

SystemAction = Literal[
    "NO_RESPONSE",
    "ACK_MINIMAL",
    "ALLOW_PROGRESS",
    "RESET_CONTEXT",
    "SUPPRESS",
]

NO_RESPONSE: Final[SystemAction] = "NO_RESPONSE"
ACK_MINIMAL: Final[SystemAction] = "ACK_MINIMAL"
ALLOW_PROGRESS: Final[SystemAction] = "ALLOW_PROGRESS"
RESET_CONTEXT: Final[SystemAction] = "RESET_CONTEXT"
SUPPRESS: Final[SystemAction] = "SUPPRESS"

SYSTEM_ACTIONS: tuple[SystemAction, ...] = (
    NO_RESPONSE,
    ACK_MINIMAL,
    ALLOW_PROGRESS,
    RESET_CONTEXT,
    SUPPRESS,
)


# =====================
# Outcomes (learning / adaptation)
# =====================

Outcome = Literal[
    "NONE",
    "ESCALATED",
    "CONTINUED",
]

NONE: Final[Outcome] = "NONE"
ESCALATED: Final[Outcome] = "ESCALATED"
CONTINUED: Final[Outcome] = "CONTINUED"

OUTCOMES: tuple[Outcome, ...] = (
    NONE,
    ESCALATED,
    CONTINUED,
)


# =====================
# Activation
# =====================

Activation = float  # 0.0 – 1.0
