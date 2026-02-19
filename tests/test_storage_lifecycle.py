# tests/test_storage_lifecycle.py
"""Suite 1: Storage / lifecycle – one blob per chat_id, persist, delete."""

import json
import logging
from pathlib import Path

from heimdall.core.classifier import Classifier
from heimdall.core.config import HeimdallConfig
from heimdall.core.decision import decide
from heimdall.core.dwell import DwellState, LabelDwell
from heimdall.core.embedder import Embedder
from heimdall.core.router import route
from heimdall.core.types import LABEL_TO_ID, REQUEST

logger = logging.getLogger(__name__)


def test_pipeline_creates_chat_files_without_explicit_persist(tmp_path: Path) -> None:
    """
    Pipeline (predict + dwell.apply) must create delta.json, prototypes.json, and dwell.json
    without the caller calling persist(). This is how embedded use (e.g. Cog) works.
    Regresses if auto-persist is removed from Classifier.predict() or LabelDwell.apply().
    """
    config = HeimdallConfig(state_dir=tmp_path / ".heimdall")
    embedder = Embedder()
    clf = Classifier(config=config, chat_id="embedded_chat")
    dwell = LabelDwell(config=config, chat_id=clf.chat_id)

    # Run one turn: no explicit persist() anywhere
    vec = embedder.encode("lets discuss auth")
    pred = clf.predict(vec)
    dwell_label = dwell.apply(pred.label, pred.activation)
    decide(dwell_label, pred.confidence, confidence_threshold=config.confidence_threshold)
    route(dwell_label)

    chat_dir = config.chat_dir(clf.chat_id)
    assert (chat_dir / "delta.json").exists(), "Classifier.predict() should auto-persist delta.json"
    assert (chat_dir / "prototypes.json").exists(), "Classifier.predict() should auto-persist prototypes.json"
    assert (chat_dir / "dwell.json").exists(), "LabelDwell.apply() should auto-persist dwell.json"

    with open(chat_dir / "delta.json") as f:
        assert isinstance(json.load(f), list)
    with open(chat_dir / "prototypes.json") as f:
        assert isinstance(json.load(f), dict)
    with open(chat_dir / "dwell.json") as f:
        data = json.load(f)
    assert "state" in data


def test_one_blob_per_chat_id_classifier(tmp_path: Path) -> None:
    """1.1: Two different chat_ids produce two directories with delta.json and prototypes.json."""
    config = HeimdallConfig(state_dir=tmp_path / ".heimdall")
    clf_a = Classifier(config=config, chat_id="chat_a")
    clf_b = Classifier(config=config, chat_id="chat_b")
    clf_a.persist()
    clf_b.persist()

    dir_a = config.chat_dir("chat_a")
    dir_b = config.chat_dir("chat_b")
    assert dir_a.exists() and dir_a.is_dir()
    assert dir_b.exists() and dir_b.is_dir()
    assert dir_a != dir_b
    assert (dir_a / "delta.json").exists()
    assert (dir_a / "prototypes.json").exists()
    assert (dir_b / "delta.json").exists()
    assert (dir_b / "prototypes.json").exists()


def test_one_blob_per_chat_id_dwell(tmp_path: Path) -> None:
    """1.2: Two LabelDwell instances (different chat_ids) each persist only to their own chat_dir."""
    config = HeimdallConfig(state_dir=tmp_path / ".heimdall")
    dwell_a = LabelDwell(config=config, chat_id="dwell_a")
    dwell_b = LabelDwell(config=config, chat_id="dwell_b")
    dwell_a.apply(REQUEST, 0.9)  # move to INTENT
    dwell_b.apply(REQUEST, 0.9)
    dwell_a.persist()
    dwell_b.persist()

    dir_a = config.chat_dir("dwell_a")
    dir_b = config.chat_dir("dwell_b")
    assert (dir_a / "dwell.json").exists()
    assert (dir_b / "dwell.json").exists()
    with open(dir_a / "dwell.json") as f:
        data_a = json.load(f)
    with open(dir_b / "dwell.json") as f:
        data_b = json.load(f)
    assert data_a["state"] == "INTENT"
    assert data_b["state"] == "INTENT"
    assert dir_a != dir_b


def test_caller_persists_when_they_want(tmp_path: Path) -> None:
    """1.3: After update_bias / maybe_add_prototype and persist(), chat dir reflects updates."""
    config = HeimdallConfig(state_dir=tmp_path / ".heimdall")
    clf = Classifier(config=config, chat_id="persist_me")
    clf.update_bias(LABEL_TO_ID[REQUEST], 0.1)
    embedder = Embedder()
    vec = embedder.encode("help with auth")
    pred = clf.predict(vec)
    clf.maybe_add_prototype(pred.label, vec, pred.confidence)
    clf.persist()

    delta_path = config.chat_dir("persist_me") / "delta.json"
    proto_path = config.chat_dir("persist_me") / "prototypes.json"
    assert delta_path.exists()
    assert proto_path.exists()
    with open(delta_path) as f:
        bias = json.load(f)
    assert isinstance(bias, list)
    assert any(b != 0 for b in bias)
    with open(proto_path) as f:
        protos = json.load(f)
    assert isinstance(protos, dict)


def test_delete_when_chat_closed(tmp_path: Path) -> None:
    """1.4: After delete_chat_state(), chat dir is gone; new instance loads fresh state."""
    config = HeimdallConfig(state_dir=tmp_path / ".heimdall")
    clf = Classifier(config=config, chat_id="to_delete")
    clf.update_bias(LABEL_TO_ID[REQUEST], 0.05)
    clf.persist()
    dwell = LabelDwell(config=config, chat_id="to_delete")
    dwell.apply(REQUEST, 0.9)
    dwell.persist()

    assert config.chat_dir("to_delete").exists()
    config.delete_chat_state("to_delete")
    assert not config.chat_dir("to_delete").exists()

    clf2 = Classifier(config=config, chat_id="to_delete")
    dwell2 = LabelDwell(config=config, chat_id="to_delete")
    assert (clf2._bias == 0).all()
    assert len(clf2.user_prototypes.store) == 0
    assert dwell2.state == DwellState.IDLE


def test_delete_idempotent(tmp_path: Path) -> None:
    """1.5: Calling delete_chat_state again for same chat_id does not raise; dir remains absent."""
    config = HeimdallConfig(state_dir=tmp_path / ".heimdall")
    config.delete_chat_state("nonexistent")
    config.delete_chat_state("nonexistent")
    assert not config.chat_dir("nonexistent").exists()

    Classifier(config=config, chat_id="twice").persist()
    assert config.chat_dir("twice").exists()
    config.delete_chat_state("twice")
    config.delete_chat_state("twice")
    assert not config.chat_dir("twice").exists()
