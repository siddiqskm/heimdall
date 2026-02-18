# tests/test_normalization_regression.py

import logging
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

logger = logging.getLogger(__name__)

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

    config = HeimdallConfig(state_dir=tmp_path / ".heimdall")

    embedder = Embedder()
    clf = Classifier(config=config)
    dwell = LabelDwell(config=config, chat_id=clf.chat_id)

    for index, (text, expected_label) in enumerate(CHAT):
        vector = embedder.encode(text)
        pred = clf.predict(vector, text=text)
        dwell_label = dwell.apply(pred.label, pred.activation)
        final_label = decide(
            dwell_label,
            pred.confidence,
            confidence_threshold=config.confidence_threshold,
        )
        logger.info(
            "%02d | %r → %s (conf=%.2f, act=%.2f)",
            index,
            text,
            final_label,
            pred.confidence,
            pred.activation,
        )
        assert final_label == expected_label
