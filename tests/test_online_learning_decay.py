# tests/test_online_learning_decay.py

import logging
from pathlib import Path

from heimdall.core.classifier import Classifier
from heimdall.core.config import HeimdallConfig
from heimdall.core.decision import decide
from heimdall.core.dwell import LabelDwell
from heimdall.core.embedder import Embedder
from heimdall.core.types import LABEL_TO_ID, REQUEST

logger = logging.getLogger(__name__)


def test_online_learning_bias_decays_without_reinforcement(tmp_path: Path) -> None:
    """
    Verifies that:
    1. Bias reinforcement increases confidence.
    2. Bias decays over time without reinforcement.
    """

    config = HeimdallConfig(state_dir=tmp_path / "heimdall_state")

    embedder = Embedder()
    clf = Classifier(config=config)
    dwell = LabelDwell(config=config, chat_id=clf.chat_id)

    text = "hadoop"
    vec = embedder.encode(text)

    # -------------------------
    # Step 1: baseline
    # -------------------------
    pred1 = clf.predict(vec)
    label1 = decide(
        dwell.apply(pred1.label, pred1.activation),
        pred1.confidence,
        confidence_threshold=config.confidence_threshold,
    )
    conf1 = pred1.confidence
    logger.info("[baseline] label=%s conf=%.2f", label1, conf1)

    assert label1 is not None

    # -------------------------
    # Step 2: reinforce REQUEST
    # -------------------------
    request_index = LABEL_TO_ID[REQUEST]
    for _ in range(8):
        clf.update_bias(request_index, delta=0.04)

    pred2 = clf.predict(vec)
    label2 = decide(
        dwell.apply(pred2.label, pred2.activation),
        pred2.confidence,
        confidence_threshold=config.confidence_threshold,
    )
    conf2 = pred2.confidence
    logger.info("[after learning] label=%s conf=%.2f", label2, conf2)

    assert label2 == REQUEST
    assert conf2 > conf1, (conf1, conf2)

    # -------------------------
    # Step 3: apply decay explicitly
    # -------------------------
    for _ in range(40):
        clf._apply_decay()

    pred3 = clf.predict(vec)
    label3 = decide(
        dwell.apply(pred3.label, pred3.activation),
        pred3.confidence,
        confidence_threshold=config.confidence_threshold,
    )
    conf3 = pred3.confidence
    logger.info("[after decay] label=%s conf=%.2f", label3, conf3)

    # Confidence must fall back
    assert conf3 < conf2, (conf2, conf3)
