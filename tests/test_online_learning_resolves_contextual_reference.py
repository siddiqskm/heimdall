# tests/test_online_learning_resolves_contextual_reference.py

from pathlib import Path

from heimdall.core.classifier import Classifier
from heimdall.core.config import HeimdallConfig
from heimdall.core.decision import decide
from heimdall.core.dwell import LabelDwell
from heimdall.core.embedder import Embedder
from heimdall.core.types import LABEL_TO_ID, REQUEST


def test_online_learning_reclassifies_weak_intent(tmp_path: Path) -> None:
    """
    Verifies that sustained strong intent shifts bias such that
    a previously weak / ambiguous phrase is reclassified as REQUEST.
    """

    config = HeimdallConfig(state_dir=tmp_path / "heimdall_state")

    embedder = Embedder()
    clf = Classifier(config=config)
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
    predicted_w1, conf_w1, act_w1 = clf.predict(vec_weak, user)
    label_w1 = decide(dwell.apply(user, predicted_w1, act_w1), conf_w1)

    print(f"[before learning] weak → {label_w1} (conf={conf_w1:.2f})")

    # MUST NOT already be REQUEST
    assert label_w1 != REQUEST

    # --------------------------------------------------
    # Step 2: sustained strong intent (simulate progress)
    # --------------------------------------------------
    request_index = LABEL_TO_ID[REQUEST]

    for _ in range(6):
        predicted_s, conf_s, act_s = clf.predict(vec_strong, user)
        label_s = decide(dwell.apply(user, predicted_s, act_s), conf_s)

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
    predicted_w2, conf_w2, act_w2 = clf.predict(vec_weak, user)
    label_w2 = decide(dwell.apply(user, predicted_w2, act_w2), conf_w2)

    print(f"[after learning] weak → {label_w2} (conf={conf_w2:.2f})")

    # THIS is the real proof of online learning
    assert label_w2 == REQUEST
    assert conf_w2 >= conf_w1
