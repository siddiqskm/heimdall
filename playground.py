# playground.py

import logging
import string
import time
from pathlib import Path

from heimdall.adapt.outcome import infer_outcome
from heimdall.core.classifier import Classifier
from heimdall.core.config import HeimdallConfig
from heimdall.core.decision import decide
from heimdall.core.dwell import LabelDwell
from heimdall.core.embedder import Embedder
from heimdall.core.learning_gate import LearningGate
from heimdall.core.router import route
from heimdall.core.types import (
    ESCALATED,
    LABEL_TO_ID,
    NO_RESPONSE,
    NONE,
    SILENT,
    Label,
    Outcome,
    SystemAction,
)

logger = logging.getLogger("heimdall.playground")

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

STATE_DIR = Path(".playground_state")
PERSIST_INTERVAL_SECONDS = 10

config = HeimdallConfig(state_dir=STATE_DIR)

embedder: Embedder = Embedder()
clf: Classifier = Classifier(config=config)
dwell = LabelDwell(config=config, chat_id=clf.chat_id)
learning_gate = LearningGate(config=config)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _is_garbage(text: str) -> bool:
    if len(text) <= 2 and all(c in string.punctuation for c in text):
        return True

    return len(text) > 2 and all(c in string.punctuation for c in text)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = " ".join(text.split())
    text = text.strip(string.punctuation)
    return text


# ------------------------------------------------------------------
# Main Loop
# ------------------------------------------------------------------

def main() -> None:
    prev_action: SystemAction | None = None
    prev_label: Label | None = None

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
        pred = clf.predict(vec, text=text)
        predicted, conf, activation = pred.label, pred.confidence, pred.activation

        dwell_label = dwell.apply(predicted, activation)
        final: Label = decide(
            dwell_label,
            conf,
            confidence_threshold=config.confidence_threshold,
        )

        action: SystemAction = route(final)

        # ---- minimal playground affordance (UX only) ----
        if final == SILENT:
            logger.info("[%s | %.2f] → %s", final, conf, NO_RESPONSE)
            logger.info("→ What would you like help with?")
        else:
            logger.info("[%s | %.2f] → %s", final, conf, action)

        # ---- outcome inference (updated) ----
        outcome: Outcome = NONE
        if prev_label is not None:
            outcome = infer_outcome(
                confidence=conf,
                next_input=text,
                confidence_threshold=config.outcome_escalated_confidence_threshold,
                min_next_len=config.outcome_escalated_min_next_len,
            )

        # ---- learning gate (single authority) ----
        if (
            prev_label is not None
            and not _is_garbage(text)
            and learning_gate.allow(
                chat_id=clf.chat_id,
                final_label=prev_label,
                confidence=conf,
                stable_turns=dwell.stable_turns(prev_label),
                action=prev_action or action,
                outcome=outcome,
                now=now,
            )
        ):
            delta = -0.05 if outcome == ESCALATED else 0.02
            clf.update_bias(LABEL_TO_ID[prev_label], delta=delta)

        # ---- persist periodically ----
        if now - last_persist > PERSIST_INTERVAL_SECONDS:
            clf.persist()
            dwell.persist()
            last_persist = now

        prev_action = action
        prev_label = final

    clf.persist()
    dwell.persist()


if __name__ == "__main__":
    import heimdall

    heimdall.configure_logging(level=logging.INFO)
    main()
