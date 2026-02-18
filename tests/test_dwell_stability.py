# tests/test_dwell_stability.py

import logging
from pathlib import Path

from heimdall.core.classifier import Classifier
from heimdall.core.config import HeimdallConfig
from heimdall.core.decision import decide
from heimdall.core.dwell import LabelDwell
from heimdall.core.embedder import Embedder
from heimdall.core.types import REQUEST

logger = logging.getLogger(__name__)


def test_request_does_not_break_on_acknowledgements_after_learning(tmp_path: Path):
    """
    Once a REQUEST intent stabilizes, lightweight acknowledgements
    should not break the dwell state.
    """

    config = HeimdallConfig(state_dir=tmp_path / ".heimdall")

    embedder = Embedder()
    clf = Classifier(config=config)
    dwell = LabelDwell(config=config, chat_id=clf.chat_id)

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
        pred = clf.predict(vec)
        dwell_label = dwell.apply(pred.label, pred.activation)
        final_label = decide(
            dwell_label,
            pred.confidence,
            confidence_threshold=config.confidence_threshold,
        )
        labels.append(final_label)
        logger.info("%r → %s", text, final_label)

    # Once REQUEST appears, it must persist
    assert REQUEST in labels

    first_request_index = labels.index(REQUEST)

    for label in labels[first_request_index:]:
        assert label == REQUEST
