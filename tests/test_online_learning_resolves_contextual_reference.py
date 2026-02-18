# tests/test_online_learning_resolves_contextual_reference.py

from pathlib import Path

from heimdall.core.classifier import Classifier
from heimdall.core.config import HeimdallConfig
from heimdall.core.decision import decide
from heimdall.core.dwell import LabelDwell
from heimdall.core.embedder import Embedder
from heimdall.core.types import LABEL_TO_ID, REQUEST


def test_online_learning_strengthens_request_bias(tmp_path: Path) -> None:
    """
    Verifies that sustained strong intent increases REQUEST confidence
    for an ambiguous phrase — regardless of its initial label.
    """

    config = HeimdallConfig(state_dir=tmp_path / "heimdall_state")

    embedder = Embedder()
    clf = Classifier(config=config)
    dwell = LabelDwell(config=config)

    user = "learning_user_reclassify"

    clf.reset_user(user)

    weak = "this part"
    strong = "this part handles user authentication"

    vec_weak = embedder.encode(weak)
    vec_strong = embedder.encode(strong)

    request_index = LABEL_TO_ID[REQUEST]

    # --------------------------------------------------
    # BEFORE learning
    # --------------------------------------------------
    predicted_w1, conf_w1, act_w1 = clf.predict(vec_weak, user)
    label_w1 = decide(
        dwell.apply(user, predicted_w1, act_w1),
        conf_w1,
        confidence_threshold=config.confidence_threshold,
    )

    print(f"[before learning] weak → {label_w1} (conf={conf_w1:.3f})")

    # --------------------------------------------------
    # Simulate sustained strong REQUEST signals
    # --------------------------------------------------
    for _ in range(6):
        predicted_s, conf_s, act_s = clf.predict(vec_strong, user)
        label_s = decide(
            dwell.apply(user, predicted_s, act_s),
            conf_s,
            confidence_threshold=config.confidence_threshold,
        )

        assert label_s == REQUEST

        clf.update_bias(
            user_id=user,
            label_index=request_index,
            delta=0.02,
        )

    clf.persist()

    # --------------------------------------------------
    # AFTER learning
    # --------------------------------------------------
    predicted_w2, conf_w2, act_w2 = clf.predict(vec_weak, user)
    label_w2 = decide(
        dwell.apply(user, predicted_w2, act_w2),
        conf_w2,
        confidence_threshold=config.confidence_threshold,
    )

    print(f"[after learning] weak → {label_w2} (conf={conf_w2:.3f})")

    # --------------------------------------------------
    # Robust assertions
    # --------------------------------------------------

    # 1. REQUEST confidence must increase
    assert conf_w2 > conf_w1

    # 2. If it was already REQUEST, it must get stronger
    if label_w1 == REQUEST:
        assert conf_w2 > conf_w1

    # 3. If it was not REQUEST, it should now become REQUEST
    else:
        assert label_w2 == REQUEST

