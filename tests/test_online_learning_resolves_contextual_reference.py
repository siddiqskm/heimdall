# tests/test_online_learning_resolves_contextual_reference.py

import logging
from pathlib import Path

from heimdall.core.classifier import Classifier
from heimdall.core.config import HeimdallConfig
from heimdall.core.decision import decide
from heimdall.core.dwell import LabelDwell
from heimdall.core.embedder import Embedder
from heimdall.core.types import LABEL_TO_ID, REQUEST

logger = logging.getLogger(__name__)


def test_online_learning_strengthens_request_bias(tmp_path: Path) -> None:
    """
    Verifies that sustained strong intent increases REQUEST confidence
    for an ambiguous phrase — regardless of its initial label.
    """

    config = HeimdallConfig(state_dir=tmp_path / "heimdall_state")

    embedder = Embedder()
    clf = Classifier(config=config)
    dwell = LabelDwell(config=config, chat_id=clf.chat_id)

    clf.reset_chat()

    weak = "this part"
    strong = "this part handles user authentication"

    vec_weak = embedder.encode(weak)
    vec_strong = embedder.encode(strong)

    request_index = LABEL_TO_ID[REQUEST]

    # --------------------------------------------------
    # BEFORE learning
    # --------------------------------------------------
    pred_w1 = clf.predict(vec_weak)
    label_w1 = decide(
        dwell.apply(pred_w1.label, pred_w1.activation),
        pred_w1.confidence,
        confidence_threshold=config.confidence_threshold,
    )
    conf_w1 = pred_w1.confidence
    logger.info("[before learning] weak → %s (conf=%.3f)", label_w1, conf_w1)

    # --------------------------------------------------
    # Simulate sustained strong REQUEST signals
    # --------------------------------------------------
    for _ in range(6):
        pred_s = clf.predict(vec_strong)
        label_s = decide(
            dwell.apply(pred_s.label, pred_s.activation),
            pred_s.confidence,
            confidence_threshold=config.confidence_threshold,
        )
        assert label_s == REQUEST
        clf.update_bias(request_index, delta=0.02)

    clf.persist()

    # --------------------------------------------------
    # AFTER learning
    # --------------------------------------------------
    pred_w2 = clf.predict(vec_weak)
    label_w2 = decide(
        dwell.apply(pred_w2.label, pred_w2.activation),
        pred_w2.confidence,
        confidence_threshold=config.confidence_threshold,
    )
    conf_w2 = pred_w2.confidence

    logger.info("[after learning] weak → %s (conf=%.3f)", label_w2, conf_w2)

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

