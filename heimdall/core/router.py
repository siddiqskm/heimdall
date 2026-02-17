# core/router.py

from collections.abc import Mapping

from heimdall.core.types import HOSTILE, REQUEST, SILENT, TOPIC_RESET, Label, SystemAction

_ROUTE_TABLE: Mapping[Label, SystemAction] = {
    SILENT: "NO_RESPONSE",
    REQUEST: "ALLOW_PROGRESS",
    TOPIC_RESET: "RESET_CONTEXT",
    HOSTILE: "SUPPRESS",
}


def route(label: Label) -> SystemAction:
    return _ROUTE_TABLE[label]
