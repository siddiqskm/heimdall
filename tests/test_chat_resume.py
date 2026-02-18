# tests/test_chat_resume.py
"""Suite 2: Resume – same chat_id loads persisted state."""

import logging
from pathlib import Path

from heimdall.core.classifier import Classifier
from heimdall.core.config import HeimdallConfig
from heimdall.core.dwell import DwellState, LabelDwell
from heimdall.core.embedder import Embedder
from heimdall.core.types import LABEL_TO_ID, REQUEST

logger = logging.getLogger(__name__)


def test_classifier_resume_new_instance(tmp_path: Path) -> None:
    """2.1: New Classifier with same chat_id loads bias/prototypes; predictions reflect them."""
    config = HeimdallConfig(state_dir=tmp_path / ".heimdall")
    embedder = Embedder()
    text = "help with backend"
    vec = embedder.encode(text)

    clf1 = Classifier(config=config, chat_id="resume_me")
    pred1 = clf1.predict(vec)
    clf1.update_bias(LABEL_TO_ID[REQUEST], 0.08)
    clf1.maybe_add_prototype(REQUEST, vec, 0.7)
    clf1.persist()

    clf2 = Classifier(config=config, chat_id="resume_me")
    pred2 = clf2.predict(vec)
    assert pred2.confidence >= pred1.confidence or pred2.label == REQUEST
    assert (clf2._bias != 0).any()


def test_classifier_resume_same_process(tmp_path: Path) -> None:
    """2.2: Two instances same chat_id in same process; second sees first's persisted state."""
    config = HeimdallConfig(state_dir=tmp_path / ".heimdall")
    embedder = Embedder()
    vec = embedder.encode("auth system")

    clf1 = Classifier(config=config, chat_id="same_process")
    clf1.update_bias(LABEL_TO_ID[REQUEST], 0.06)
    clf1.persist()

    clf2 = Classifier(config=config, chat_id="same_process")
    pred2 = clf2.predict(vec)
    assert pred2.label == REQUEST or pred2.confidence > 0.35


def test_dwell_resume(tmp_path: Path) -> None:
    """2.3: New LabelDwell with same chat_id loads FSM state (e.g. INTENT)."""
    config = HeimdallConfig(state_dir=tmp_path / ".heimdall")
    dwell1 = LabelDwell(config=config, chat_id="resume_dwell")
    dwell1.apply(REQUEST, 0.85)
    dwell1.apply(REQUEST, 0.8)
    assert dwell1.state == DwellState.INTENT
    dwell1.persist()

    dwell2 = LabelDwell(config=config, chat_id="resume_dwell")
    assert dwell2.state == DwellState.INTENT
    out = dwell2.apply(REQUEST, 0.9)
    assert out == REQUEST
    assert dwell2.state == DwellState.INTENT
