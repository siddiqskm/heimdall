# tests/test_new_chat.py
"""Suite 3: New chat – fresh state, generated ids, reset_chat."""

import logging
from pathlib import Path

from heimdall.core.classifier import Classifier
from heimdall.core.config import HeimdallConfig
from heimdall.core.dwell import DwellState, LabelDwell
from heimdall.core.types import LABEL_TO_ID, REQUEST

logger = logging.getLogger(__name__)


def test_new_chat_generates_id(tmp_path: Path) -> None:
    """3.1: Classifier() and Classifier(chat_id=None) get non-empty chat_id; two instances differ."""
    config = HeimdallConfig(state_dir=tmp_path / "state")
    clf1 = Classifier(config=config)
    clf2 = Classifier(config=config)
    clf3 = Classifier(config=config, chat_id=None)

    assert clf1.chat_id
    assert clf2.chat_id
    assert clf3.chat_id
    assert clf1.chat_id != clf2.chat_id
    assert len(clf1.chat_id) >= 16

    dwell1 = LabelDwell(config=config)
    dwell2 = LabelDwell(config=config)
    assert dwell1.chat_id != dwell2.chat_id


def test_new_chat_unused_id_fresh_state(tmp_path: Path) -> None:
    """3.2: Classifier/Dwell with brand-new chat_id have zero bias and IDLE."""
    config = HeimdallConfig(state_dir=tmp_path / "state")
    clf = Classifier(config=config, chat_id="brand_new_123")
    assert (clf._bias == 0).all()
    assert len(clf.user_prototypes.store) == 0

    dwell = LabelDwell(config=config, chat_id="brand_new_456")
    assert dwell.state == DwellState.IDLE
    out = dwell.apply(REQUEST, 0.9)
    assert out == REQUEST
    assert dwell.state == DwellState.INTENT


def test_reset_chat_clears_and_persists(tmp_path: Path) -> None:
    """3.3: reset_chat() clears in-memory and on-disk state; new instance loads reset state."""
    config = HeimdallConfig(state_dir=tmp_path / "state")
    clf = Classifier(config=config, chat_id="reset_me")
    clf.update_bias(LABEL_TO_ID[REQUEST], 0.1)
    clf.persist()
    clf.reset_chat()

    assert (clf._bias == 0).all()
    assert len(clf.user_prototypes.store) == 0

    clf2 = Classifier(config=config, chat_id="reset_me")
    assert (clf2._bias == 0).all()
    assert len(clf2.user_prototypes.store) == 0
