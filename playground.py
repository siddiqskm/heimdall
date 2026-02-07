# playground.py

from core.dwell import LabelDwell
from core.embedder import Embedder
from core.classifier import Classifier
from core.decision import decide
from core.router import route
from core.types import (
    Label,
    SystemAction,
    LABEL_TO_ID,
    Outcome,
    NONE,
    ESCALATED,
    SILENT,
    NO_RESPONSE,
)
from adapt.outcome import infer_outcome
from core.learning_gate import LearningGate

import time
import string


embedder: Embedder = Embedder()
clf: Classifier = Classifier("models/lr.joblib")
dwell = LabelDwell()
learning_gate = LearningGate()

user_id: str = "test_user"


def _is_garbage(text: str) -> bool:
    """
    Very small local filter to avoid learning from keyboard mash.
    This does NOT affect classification, only learning.
    """
    if len(text) <= 2:
        return True
    if all(c in string.punctuation for c in text):
        return True
    return False


def normalize_text(text: str) -> str:
    """
    Lightweight semantic-preserving normalization.
    Applied BEFORE embedding.
    """
    text = text.lower()
    text = text.strip()

    # collapse whitespace
    text = " ".join(text.split())

    # strip surrounding punctuation only
    text = text.strip(string.punctuation)

    return text


def main() -> None:
    prev_action: SystemAction | None = None
    prev_label: Label | None = None
    prev_time: float | None = None

    last_persist: float = time.time()

    while True:
        try:
            raw_text: str = input("> ").strip()
            if not raw_text:
                continue

            text = normalize_text(raw_text)
        except EOFError:
            break

        if not text:
            continue

        now = time.time()

        # ---- classification ----
        vec = embedder.encode(text)
        predicted, conf, activation = clf.predict(vec, user_id)
        dwell_label = dwell.apply(user_id, predicted, activation)
        final: Label = decide(dwell_label, conf)

        action: SystemAction = route(final)

        # ---- minimal playground affordance (UX only) ----
        if final is SILENT:
            print(f"[{final} | {conf:.2f}] → {NO_RESPONSE}")
            print("→ What would you like help with?")
        else:
            print(f"[{final} | {conf:.2f}] → {action}")

        # ---- outcome inference ----
        outcome: Outcome = NONE
        if prev_action is not None and prev_time is not None:
            outcome = infer_outcome(
                prev_action=prev_action,
                next_input=text,
                time_gap=now - prev_time,
            )

        # ---- learning gate (single authority) ----
        if (
            prev_label is not None
            and not _is_garbage(text)
            and learning_gate.allow(
                user_id=user_id,
                final_label=prev_label,
                confidence=conf,
                stable_turns=dwell.stable_turns(user_id, prev_label),
                action=prev_action or action,
                outcome=outcome,
                now=now,
            )
        ):
            delta = -0.05 if outcome is ESCALATED else 0.02
            clf.update_bias(
                user_id,
                LABEL_TO_ID[prev_label],
                delta=delta,
            )

        # ---- persist periodically ----
        if now - last_persist > 10:
            clf.persist()
            last_persist = now

        prev_action = action
        prev_label = final
        prev_time = now

    clf.persist()


if __name__ == "__main__":
    main()
