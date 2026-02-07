# tests/test_dwell_stability.py

from core.classifier import Classifier
from core.embedder import Embedder
from core.dwell import LabelDwell
from core.decision import decide
from core.types import REQUEST


def test_request_does_not_break_on_acknowledgements_after_learning():
    embedder = Embedder()
    clf = Classifier("models/lr.joblib")
    dwell = LabelDwell()

    user = "dwell_user"

    sequence = [
        "awesome",
        "cool",
        "need help with backend",
        "awesome",
        "cool",
        "nice",
        "right",
        "auth system",
    ]

    labels = []

    for text in sequence:
        vec = embedder.encode(text)
        label, conf, act = clf.predict(vec, user)
        label = decide(dwell.apply(user, label, act), conf)
        labels.append(label)

        print(f"{text!r} → {label}")

    # Once REQUEST appears, it must persist
    assert REQUEST in labels
    first_request = labels.index(REQUEST)

    for l in labels[first_request:]:
        assert l == REQUEST
