# tests/test_long_session_stability.py

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


def test_long_session_stability(tmp_path: Path) -> None:
    """
    Stress test 200+ turns to ensure:
    - Reset always fires
    - Hostile cooldown recovers
    - Intent re-entry works
    - No drift or permanent state corruption
    """

    config = HeimdallConfig(state_dir=tmp_path / "heimdall_state")

    embedder = Embedder()
    clf = Classifier(config=config)
    dwell = LabelDwell(config=config, chat_id=clf.chat_id, debug=True)

    # Structured repeating pattern
    sequence: list[tuple[str, Label]] = [
        # idle start
        ("hey", SILENT),

        # intent entry
        ("lets discuss travel", REQUEST),
        ("thats interesting", REQUEST),
        ("cool", REQUEST),

        # reset
        ("lets switch gears", TOPIC_RESET),

        # post-reset silence
        ("cool", SILENT),

        # new intent
        ("lets discuss tech", REQUEST),
        ("ai is interesting", REQUEST),

        # hostility spike
        ("what the hell", HOSTILE),
        ("no this is stupid", HOSTILE),

        # recovery
        ("anyways lets continue", REQUEST),
        ("cool", REQUEST),
    ]

    # Repeat pattern to exceed 200 turns
    full_sequence = sequence * 20  # 12 * 20 = 240 turns

    hostile_count = 0
    reset_count = 0

    for idx, (text, expected_label) in enumerate(full_sequence):
        vector = embedder.encode(text)
        pred = clf.predict(vector)
        dwell_label = dwell.apply(pred.label, pred.activation)
        final_label = decide(
            dwell_label,
            pred.confidence,
            confidence_threshold=config.confidence_threshold,
        )
        if final_label == HOSTILE:
            hostile_count += 1
        if final_label == TOPIC_RESET:
            reset_count += 1
        logger.info(
            "%03d | %-25r pred=%-10s dwell=%-10s final=%-10s (conf=%.2f, act=%.2f)",
            idx,
            text,
            pred.label,
            dwell_label,
            final_label,
            pred.confidence,
            pred.activation,
        )

        assert final_label == expected_label, (
            f"Drift detected at turn {idx}: "
            f"{text!r} expected {expected_label} "
            f"got {final_label}"
        )

    # Sanity: ensure hostile + reset fired repeatedly
    assert hostile_count >= 20
    assert reset_count >= 20
