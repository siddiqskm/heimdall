# tests/test_long_session_stability.py

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
    dwell = LabelDwell(debug=True)

    user_id = "long_session_user"

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

        predicted, confidence, activation = clf.predict(vector, user_id)
        dwell_label = dwell.apply(user_id, predicted, activation)
        final_label = decide(dwell_label, confidence)

        # Track events
        if final_label == HOSTILE:
            hostile_count += 1
        if final_label == TOPIC_RESET:
            reset_count += 1

        print(
            f"{idx:03d} | {text!r:<25} "
            f"pred={predicted:<10} "
            f"dwell={dwell_label:<10} "
            f"final={final_label:<10} "
            f"(conf={confidence:.2f}, act={activation:.2f})"
        )

        assert final_label == expected_label, (
            f"Drift detected at turn {idx}: "
            f"{text!r} expected {expected_label} "
            f"got {final_label}"
        )

    # Sanity: ensure hostile + reset fired repeatedly
    assert hostile_count >= 20
    assert reset_count >= 20
