# tests/test_baseline.py

from core.classifier import Classifier
from core.embedder import Embedder
from core.decision import decide
from core.dwell import LabelDwell


TEST_CASES = [
    # --- silence / noise ---
    ("", "SILENT"),
    ("hmm", "SILENT"),
    ("uh", "SILENT"),
    ("...", "SILENT"),

    # --- intent entry ---
    ("backend", "REQUEST"),
    ("auth system", "REQUEST"),
    ("lets build auth", "REQUEST"),

    # --- dwell protection (ack should NOT break intent) ---
    ("awesome", "REQUEST"),
    ("cool", "REQUEST"),
    ("okay", "REQUEST"),
    ("yup", "REQUEST"),

    # --- meaningful continuation ---
    ("login flow", "REQUEST"),
    ("session cookie", "REQUEST"),

    # --- acknowledgement still inside intent ---
    ("nice", "REQUEST"),

    # --- topic reset ---
    ("change topic", "TOPIC_RESET"),
    ("lets talk about something else", "TOPIC_RESET"),

    # --- hostility ---
    ("wtf", "HOSTILE"),
    ("go to hell", "HOSTILE"),
]


def test_baseline() -> None:
    embedder = Embedder()
    clf = Classifier("models/lr.joblib")
    dwell = LabelDwell()

    user_id = "baseline_user"

    for text, expected in TEST_CASES:
        vector = embedder.encode(text)

        # classifier returns (label, confidence, activation)
        label, confidence, activation = clf.predict(vector, user_id)

        # dwell uses activation (not confidence)
        label = dwell.apply(user_id, label, activation)

        # final decision gate
        label = decide(label, confidence)

        print(f"{text!r} → {label} (conf={confidence:.2f}, act={activation:.2f})")

        assert label == expected
