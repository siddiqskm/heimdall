# examples/gatekeeper_bot.py
#!/usr/bin/env python3
"""
Integration example: use heimdall as a gate in front of your assistant (LLM, API, or rule-based).

For each user message we:
  1. Run the heimdall pipeline (embed → classify → dwell → decide → route).
  2. Branch on the system action and only call the assistant when appropriate.

Run from repo root: poetry run python examples/gatekeeper_bot.py
"""

import logging
from pathlib import Path

from heimdall import (
    ALLOW_PROGRESS,
    NO_RESPONSE,
    RESET_CONTEXT,
    SUPPRESS,
    Classifier,
    Embedder,
    HeimdallConfig,
    LabelDwell,
    SystemAction,
    configure_logging,
    decide,
    route,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STATE_DIR = Path(__file__).resolve().parent.parent / ".heimdall"
config = HeimdallConfig(state_dir=STATE_DIR)
configure_logging(level=logging.INFO)

embedder = Embedder()
clf = Classifier(config=config)
dwell = LabelDwell(config=config, chat_id=clf.chat_id)


# ---------------------------------------------------------------------------
# Stub assistant (replace with your LLM/API call)
# ---------------------------------------------------------------------------

def call_assistant(user_message: str, context: list[str]) -> str:
    """Your backend: LLM, REST API, or rule-based reply."""
    # Stub: echo intent and a short reply
    return f"[Assistant] Got: {user_message!r} (context length={len(context)})"


# ---------------------------------------------------------------------------
# Gate: one turn
# ---------------------------------------------------------------------------

def gate_turn(user_message: str, context: list[str]) -> tuple[SystemAction, str | None]:
    """
    Run heimdall and return (action, reply).
    reply is None for NO_RESPONSE; otherwise a string for the user.
    """
    vec = embedder.encode(user_message)
    pred = clf.predict(vec, text=user_message)
    predicted, confidence, activation = pred.label, pred.confidence, pred.activation
    dwell_label = dwell.apply(predicted, activation)
    final_label = decide(
        dwell_label,
        confidence,
        confidence_threshold=config.confidence_threshold,
    )
    action = route(final_label)

    if action == NO_RESPONSE:
        return action, None

    if action == SUPPRESS:
        return action, "I'm here when you'd like to continue."

    if action == RESET_CONTEXT:
        return action, "Sure, what would you like to talk about?"

    if action == ALLOW_PROGRESS:
        reply = call_assistant(user_message, context)
        return action, reply

    return action, None


# ---------------------------------------------------------------------------
# Demo loop
# ---------------------------------------------------------------------------

def main() -> None:
    context: list[str] = []

    turns = [
        "hey",
        "can you help me with authentication?",
        "cool",
        "what about JWT vs sessions?",
        "change topic",
        "lets talk about something else",
        "ok",
    ]

    print("Heimdall gatekeeper integration example\n")
    for msg in turns:
        action, reply = gate_turn(msg, context)

        if action == ALLOW_PROGRESS:
            context.append(msg)
        elif action == RESET_CONTEXT:
            context.clear()

        print(f"  User: {msg!r}")
        print(f"  → {action}", end="")
        if reply is not None:
            print(f" → {reply}")
        else:
            print()
        print()

    print("Done. In a real app you'd wire gate_turn() into your HTTP handler or message loop.")


if __name__ == "__main__":
    main()
