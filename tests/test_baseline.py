# tests/test_baseline.py

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

TEST_CASES: list[tuple[str, Label]] = [
    # --- silence / noise ---
    ("", SILENT),
    ("hmm", SILENT),
    ("uh", SILENT),
    ("...", SILENT),

    # --- intent entry ---
    ("backend", REQUEST),
    ("auth system", REQUEST),
    ("lets build auth", REQUEST),

    # --- dwell protection (ack should NOT break intent) ---
    ("awesome", REQUEST),
    ("cool", REQUEST),
    ("okay", REQUEST),
    ("yup", REQUEST),

    # --- meaningful continuation ---
    ("login flow", REQUEST),
    ("session cookie", REQUEST),

    # --- acknowledgement still inside intent ---
    ("nice", REQUEST),

    # --- topic reset ---
    ("change topic", TOPIC_RESET),
    ("lets talk about something else", TOPIC_RESET),

    # --- hostility ---
    ("wtf", HOSTILE),
    ("go to hell", HOSTILE),
]


def test_baseline(tmp_path: Path) -> None:
    """
    Baseline behavioral contract for core classification + dwell + decision.
    """

    config = HeimdallConfig(state_dir=tmp_path / "heimdall_state")

    embedder = Embedder()
    clf = Classifier(config=config)
    dwell = LabelDwell(config=config, chat_id=clf.chat_id)

    for text, expected_label in TEST_CASES:
        vector = embedder.encode(text)
        pred = clf.predict(vector)
        dwell_label = dwell.apply(pred.label, pred.activation)
        final_label = decide(
            dwell_label,
            pred.confidence,
            confidence_threshold=config.confidence_threshold,
        )
        logger.info(
            "%r → %s (conf=%.2f, act=%.2f)",
            text,
            final_label,
            pred.confidence,
            pred.activation,
        )
        assert final_label == expected_label
