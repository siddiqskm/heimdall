# core/router.py

from typing import Mapping
from core.types import Label, SystemAction

_ROUTE_TABLE: Mapping[Label, SystemAction] = {
    "SILENT": "NO_RESPONSE",
    "STEER": "ACK_MINIMAL",
    "REQUEST": "ALLOW_PROGRESS",
    "TOPIC_RESET": "RESET_CONTEXT",
    "HOSTILE": "SUPPRESS",
}

def route(label: Label) -> SystemAction:
    return _ROUTE_TABLE[label]
