# tests/test_hostile_soft_recovery.py

import logging
from pathlib import Path

from heimdall.core.classifier import Classifier
from heimdall.core.config import HeimdallConfig
from heimdall.core.decision import decide
from heimdall.core.dwell import LabelDwell
from heimdall.core.embedder import Embedder
from heimdall.core.types import HOSTILE, REQUEST

logger = logging.getLogger(__name__)


def test_soft_hostile_recovery_flow(tmp_path: Path) -> None:
    """
    Ensures that:
    1. Strong intent is established
    2. Hostility overrides
    3. Hostile suppression persists briefly
    4. Productive intent eventually recovers
    """

    config = HeimdallConfig(state_dir=tmp_path / ".heimdall")

    embedder = Embedder()
    clf = Classifier(config=config)
    dwell = LabelDwell(config=config, chat_id=clf.chat_id)

    strong = "need help with aws deployment"
    vec_s = embedder.encode(strong)

    pred1 = clf.predict(vec_s)
    label1 = decide(
        dwell.apply(pred1.label, pred1.activation),
        pred1.confidence,
        confidence_threshold=config.confidence_threshold,
    )
    logger.info("[intent] label=%s conf=%.2f", label1, pred1.confidence)
    assert label1 == REQUEST

    hostile = "what the hell"
    vec_h = embedder.encode(hostile)
    pred2 = clf.predict(vec_h)
    label2 = decide(
        dwell.apply(pred2.label, pred2.activation),
        pred2.confidence,
        confidence_threshold=config.confidence_threshold,
    )
    logger.info("[hostile] label=%s conf=%.2f", label2, pred2.confidence)
    assert label2 == HOSTILE

    calm = "cool"
    vec_c = embedder.encode(calm)
    pred3 = clf.predict(vec_c)
    label3 = decide(
        dwell.apply(pred3.label, pred3.activation),
        pred3.confidence,
        confidence_threshold=config.confidence_threshold,
    )
    logger.info("[cooldown-1] label=%s conf=%.2f", label3, pred3.confidence)
    assert label3 == HOSTILE

    calm2 = "okay"
    vec_c2 = embedder.encode(calm2)
    pred4 = clf.predict(vec_c2)
    label4 = decide(
        dwell.apply(pred4.label, pred4.activation),
        pred4.confidence,
        confidence_threshold=config.confidence_threshold,
    )
    logger.info("[cooldown-2] label=%s conf=%.2f", label4, pred4.confidence)
    assert label4 == HOSTILE

    pred5 = clf.predict(vec_s)
    label5 = decide(
        dwell.apply(pred5.label, pred5.activation),
        pred5.confidence,
        confidence_threshold=config.confidence_threshold,
    )
    logger.info("[recovered] label=%s conf=%.2f", label5, pred5.confidence)
    assert label5 == REQUEST
