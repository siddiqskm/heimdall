# tests/test_baseline.py

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
    dwell = LabelDwell()

    user_id = "baseline_user"

    for text, expected_label in TEST_CASES:
        vector = embedder.encode(text)

        # classifier returns (label, confidence, activation)
        predicted, confidence, activation = clf.predict(vector, user_id)

        # dwell uses activation (not confidence)
        dwell_label = dwell.apply(user_id, predicted, activation)

        # final decision gate
        final_label = decide(dwell_label, confidence)

        print(
            f"{text!r} → {final_label} "
            f"(conf={confidence:.2f}, act={activation:.2f})"
        )

        assert final_label == expected_label
