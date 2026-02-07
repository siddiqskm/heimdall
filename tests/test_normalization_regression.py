# tests/test_conversational_flow.py

from core.classifier import Classifier
from core.embedder import Embedder
from core.decision import decide
from core.dwell import LabelDwell


CHAT = [
    # ----------------------------
    # Warm-up / social noise
    # ----------------------------
    ("hey", "SILENT"),
    ("hello", "SILENT"),
    ("hmm", "SILENT"),

    # ----------------------------
    # Exploratory intent (should NOT be silent)
    # ----------------------------
    ("any suggestions to discuss?", "REQUEST"),
    ("lets talk about textiles", "REQUEST"),

    # ----------------------------
    # Acknowledgement loop (stay in intent)
    # ----------------------------
    ("cool", "REQUEST"),
    ("awesome", "REQUEST"),
    ("nice", "REQUEST"),

    # ----------------------------
    # Concrete continuation
    # ----------------------------
    ("handloom history", "REQUEST"),
    ("traditional weaving", "REQUEST"),

    # ----------------------------
    # Topic reset (explicit)
    # ----------------------------
    ("lets switch gears", "TOPIC_RESET"),
    ("change topic", "TOPIC_RESET"),

    # ----------------------------
    # Post-reset noise (must NOT leak intent)
    # ----------------------------
    ("okay", "SILENT"),
    ("cool", "SILENT"),

    # ----------------------------
    # New intent after reset
    # ----------------------------
    ("need help with a frontend project", "REQUEST"),
    ("sveltekit basics", "REQUEST"),

    # ----------------------------
    # Frustration but still intent
    # ----------------------------
    ("this is confusing", "REQUEST"),
    ("can you simplify", "REQUEST"),

    # ----------------------------
    # Hostility overrides everything
    # ----------------------------
    ("what the hell", "HOSTILE"),
    ("go to hell", "HOSTILE"),
]


def test_realistic_conversation_flow() -> None:
    embedder = Embedder()
    clf = Classifier("models/lr.joblib")
    dwell = LabelDwell()

    user_id = "conversation_user"

    for idx, (text, expected) in enumerate(CHAT):
        vector = embedder.encode(text)

        label, confidence, activation = clf.predict(vector, user_id)
        label = dwell.apply(user_id, label, activation)
        label = decide(label, confidence)

        print(
            f"{idx:02d} | {text!r} → {label} "
            f"(conf={confidence:.2f}, act={activation:.2f})"
        )

        assert label == expected
