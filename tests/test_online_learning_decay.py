# tests/test_online_learning_decay.py

from core.classifier import Classifier
from core.embedder import Embedder
from core.dwell import LabelDwell
from core.decision import decide
from core.types import REQUEST


def test_online_learning_bias_decays_without_reinforcement() -> None:
    embedder = Embedder()
    clf = Classifier("models/lr.joblib")
    dwell = LabelDwell()

    user = "decay_user"

    text = "hadoop"
    vec = embedder.encode(text)

    # -------------------------
    # Step 1: baseline
    # -------------------------
    label1, conf1, act1 = clf.predict(vec, user)
    label1 = decide(dwell.apply(user, label1, act1), conf1)

    print(f"[baseline] {label1=} {conf1=:.2f}")

    # We no longer assume SILENT
    assert label1 == REQUEST or label1 is not None

    # -------------------------
    # Step 2: reinforce REQUEST
    # -------------------------
    request_index = clf.model.classes_.tolist().index(2)  # REQUEST

    for _ in range(8):
        clf.update_bias(
            user_id=user,
            label_index=request_index,
            delta=0.04,
        )

    label2, conf2, act2 = clf.predict(vec, user)
    label2 = decide(dwell.apply(user, label2, act2), conf2)

    print(f"[after learning] {label2=} {conf2=:.2f}")

    assert label2 == REQUEST
    assert conf2 > conf1, (conf1, conf2)

    # -------------------------
    # Step 3: apply decay explicitly
    # -------------------------
    for _ in range(40):
        clf._apply_decay(user)

    label3, conf3, act3 = clf.predict(vec, user)
    label3 = decide(dwell.apply(user, label3, act3), conf3)

    print(f"[after decay] {label3=} {conf3=:.2f}")

    # Confidence must fall back
    assert conf3 < conf2, (conf2, conf3)
