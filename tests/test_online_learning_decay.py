# tests/test_online_learning_decay.py

from pathlib import Path

from heimdall.core.classifier import Classifier
from heimdall.core.config import HeimdallConfig
from heimdall.core.decision import decide
from heimdall.core.dwell import LabelDwell
from heimdall.core.embedder import Embedder
from heimdall.core.types import LABEL_TO_ID, REQUEST


def test_online_learning_bias_decays_without_reinforcement(tmp_path: Path) -> None:
    """
    Verifies that:
    1. Bias reinforcement increases confidence.
    2. Bias decays over time without reinforcement.
    """

    config = HeimdallConfig(state_dir=tmp_path / "heimdall_state")

    embedder = Embedder()
    clf = Classifier(config=config)
    dwell = LabelDwell(config=config)

    user = "decay_user"

    text = "hadoop"
    vec = embedder.encode(text)

    # -------------------------
    # Step 1: baseline
    # -------------------------
    predicted1, conf1, act1 = clf.predict(vec, user)
    label1 = decide(
        dwell.apply(user, predicted1, act1),
        conf1,
        confidence_threshold=config.confidence_threshold,
    )

    print(f"[baseline] label={label1} conf={conf1:.2f}")

    # Do not assume initial class; only ensure valid label
    assert label1 is not None

    # -------------------------
    # Step 2: reinforce REQUEST
    # -------------------------
    request_index = LABEL_TO_ID[REQUEST]

    for _ in range(8):
        clf.update_bias(
            user_id=user,
            label_index=request_index,
            delta=0.04,
        )

    predicted2, conf2, act2 = clf.predict(vec, user)
    label2 = decide(
        dwell.apply(user, predicted2, act2),
        conf2,
        confidence_threshold=config.confidence_threshold,
    )

    print(f"[after learning] label={label2} conf={conf2:.2f}")

    assert label2 == REQUEST
    assert conf2 > conf1, (conf1, conf2)

    # -------------------------
    # Step 3: apply decay explicitly
    # -------------------------
    for _ in range(40):
        clf._apply_decay(user)

    predicted3, conf3, act3 = clf.predict(vec, user)
    label3 = decide(
        dwell.apply(user, predicted3, act3),
        conf3,
        confidence_threshold=config.confidence_threshold,
    )

    print(f"[after decay] label={label3} conf={conf3:.2f}")

    # Confidence must fall back
    assert conf3 < conf2, (conf2, conf3)
