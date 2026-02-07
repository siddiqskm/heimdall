# tests/test_online_learning_resolves_contextual_reference.py

from core.classifier import Classifier
from core.embedder import Embedder
from core.dwell import LabelDwell
from core.decision import decide
from core.types import REQUEST


def test_online_learning_reclassifies_weak_intent() -> None:
    embedder = Embedder()
    clf = Classifier("models/lr.joblib")
    dwell = LabelDwell()

    user = "learning_user_reclassify"

    # HARD RESET — test isolation
    clf.reset_user(user)

    # Weak / underspecified phrase (borderline)
    weak = "this part"
    strong = "this part handles user authentication"


    vec_weak = embedder.encode(weak)
    vec_strong = embedder.encode(strong)

    # --------------------------------------------------
    # Step 1: weak intent BEFORE learning
    # --------------------------------------------------
    label_w1, conf_w1, act_w1 = clf.predict(vec_weak, user)
    label_w1 = decide(dwell.apply(user, label_w1, act_w1), conf_w1)

    print(f"[before learning] weak → {label_w1} (conf={conf_w1:.2f})")

    # MUST NOT already be REQUEST
    assert label_w1 != REQUEST

    # --------------------------------------------------
    # Step 2: sustained strong intent (simulate progress)
    # --------------------------------------------------
    request_index = clf.model.classes_.tolist().index(2)  # REQUEST

    for _ in range(6):
        label_s, conf_s, act_s = clf.predict(vec_strong, user)
        label_s = decide(dwell.apply(user, label_s, act_s), conf_s)

        assert label_s == REQUEST

        clf.update_bias(
            user_id=user,
            label_index=request_index,
            delta=0.02,
        )

    clf.persist()

    # --------------------------------------------------
    # Step 3: weak intent AFTER learning
    # --------------------------------------------------
    label_w2, conf_w2, act_w2 = clf.predict(vec_weak, user)
    label_w2 = decide(dwell.apply(user, label_w2, act_w2), conf_w2)

    print(f"[after learning] weak → {label_w2} (conf={conf_w2:.2f})")

    # THIS is the real proof of online learning
    assert label_w2 == REQUEST
    assert conf_w2 >= conf_w1
