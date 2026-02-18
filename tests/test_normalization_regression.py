# tests/test_normalization_regression.py

from pathlib import Path

from heimdall.core.classifier import Classifier
from heimdall.core.config import HeimdallConfig
from heimdall.core.decision import decide
from heimdall.core.dwell import LabelDwell
from heimdall.core.embedder import Embedder
from heimdall.core.types import (
    HOSTILE,
    REQUEST,
    SILENT,
    TOPIC_RESET,
    Label,
)

CHAT: list[tuple[str, Label]] = [
    # ----------------------------
    # Warm-up / social noise
    # ----------------------------
    ("hey", SILENT),
    ("hello", SILENT),
    ("hmm", SILENT),

    # ----------------------------
    # Exploratory intent (should NOT be silent)
    # ----------------------------
    ("any suggestions to discuss?", REQUEST),
    ("lets talk about textiles", REQUEST),

    # ----------------------------
    # Acknowledgement loop (stay in intent)
    # ----------------------------
    ("cool", REQUEST),
    ("awesome", REQUEST),
    ("nice", REQUEST),

    # ----------------------------
    # Concrete continuation
    # ----------------------------
    ("handloom history", REQUEST),
    ("traditional weaving", REQUEST),

    # ----------------------------
    # Topic reset (explicit)
    # ----------------------------
    ("lets switch gears", TOPIC_RESET),
    ("change topic", TOPIC_RESET),

    # ----------------------------
    # Post-reset noise (must NOT leak intent)
    # ----------------------------
    ("okay", SILENT),
    ("cool", SILENT),

    # ----------------------------
    # New intent after reset
    # ----------------------------
    ("need help with a frontend project", REQUEST),
    ("sveltekit basics", REQUEST),

    # ----------------------------
    # Frustration but still intent
    # ----------------------------
    ("this is confusing", REQUEST),
    ("can you simplify", REQUEST),

    # ----------------------------
    # Hostility overrides everything
    # ----------------------------
    ("what the hell", HOSTILE),
    ("go to hell", HOSTILE),
]


def test_realistic_conversation_flow(tmp_path: Path) -> None:
    """
    Validates realistic conversational flow:

    - Social noise handling
    - Exploratory intent detection
    - Dwell stability
    - Topic reset enforcement
    - Hostility override behavior
    """

    config = HeimdallConfig(state_dir=tmp_path / "heimdall_state")

    embedder = Embedder()
    clf = Classifier(config=config)
    dwell = LabelDwell(config=config)

    user_id = "conversation_user"

    for index, (text, expected_label) in enumerate(CHAT):
        vector = embedder.encode(text)

        predicted, confidence, activation = clf.predict(vector, user_id)
        dwell_label = dwell.apply(user_id, predicted, activation)
        final_label = decide(
            dwell_label,
            confidence,
            confidence_threshold=config.confidence_threshold,
        )

        print(
            f"{index:02d} | {text!r} → {final_label} "
            f"(conf={confidence:.2f}, act={activation:.2f})"
        )

        assert final_label == expected_label
