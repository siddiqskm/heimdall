# tests/test_dwell_fsm_states.py

from pathlib import Path

import pytest

from heimdall.core.classifier import Classifier
from heimdall.core.config import HeimdallConfig
from heimdall.core.decision import decide
from heimdall.core.dwell import DwellState, LabelDwell
from heimdall.core.embedder import Embedder
from heimdall.core.types import (
    HOSTILE,
    REQUEST,
    SILENT,
    TOPIC_RESET,
    Label,
)


@pytest.fixture()
def dwell_setup(
    tmp_path: Path,
) -> tuple[Embedder, Classifier, LabelDwell]:
    """
    Creates fresh Heimdall pipeline for FSM validation (one chat per run).
    """
    config: HeimdallConfig = HeimdallConfig(
        state_dir=tmp_path / ".heimdall"
    )
    embedder: Embedder = Embedder()
    clf: Classifier = Classifier(config=config)
    dwell: LabelDwell = LabelDwell(config=config, chat_id=clf.chat_id, debug=False)
    return embedder, clf, dwell


def run_turn(
    embedder: Embedder,
    clf: Classifier,
    dwell: LabelDwell,
    text: str,
) -> tuple[Label, DwellState]:
    """Execute a single turn and return final label and FSM state."""
    vector = embedder.encode(text)
    pred = clf.predict(vector)
    predicted, confidence, activation = pred.label, pred.confidence, pred.activation
    dwell_label = dwell.apply(predicted, activation)
    final_label = decide(dwell_label, confidence)
    return final_label, dwell.state


def test_idle_to_intent_transition(
    dwell_setup: tuple[Embedder, Classifier, LabelDwell],
) -> None:
    """Validates IDLE → INTENT transition."""
    embedder, clf, dwell = dwell_setup
    label, state = run_turn(embedder, clf, dwell, "lets discuss space")

    assert label == REQUEST
    assert state == DwellState.INTENT


def test_intent_continuity_with_acknowledgement(
    dwell_setup: tuple[Embedder, Classifier, LabelDwell],
) -> None:
    """Validates benign SILENT inherits intent."""
    embedder, clf, dwell = dwell_setup
    run_turn(embedder, clf, dwell, "lets discuss space")
    label, state = run_turn(embedder, clf, dwell, "cool")

    assert label == REQUEST
    assert state == DwellState.INTENT


def test_intent_decay_on_second_silent(
    dwell_setup: tuple[Embedder, Classifier, LabelDwell],
) -> None:
    """Validates intent exits after decay threshold."""
    embedder, clf, dwell = dwell_setup
    run_turn(embedder, clf, dwell, "lets discuss space")
    run_turn(embedder, clf, dwell, "cool")
    label, state = run_turn(embedder, clf, dwell, "hey")

    assert label == REQUEST
    assert state == DwellState.INTENT


def test_hostile_entry_and_recovery(
    dwell_setup: tuple[Embedder, Classifier, LabelDwell],
) -> None:
    """Validates HOSTILE entry and cooldown-based recovery."""
    embedder, clf, dwell = dwell_setup
    label, state = run_turn(embedder, clf, dwell, "what the hell")
    assert label == HOSTILE
    assert state == DwellState.HOSTILE

    label, state = run_turn(embedder, clf, dwell, "lets continue")
    assert label == HOSTILE
    assert state == DwellState.HOSTILE

    label, state = run_turn(embedder, clf, dwell, "continue please")
    assert label == HOSTILE
    assert state == DwellState.HOSTILE

    label, state = run_turn(embedder, clf, dwell, "lets continue")
    assert label == REQUEST
    assert state == DwellState.INTENT


def test_topic_reset_transition(
    dwell_setup: tuple[Embedder, Classifier, LabelDwell],
) -> None:
    """Validates reset forces POST_RESET state."""
    embedder, clf, dwell = dwell_setup
    run_turn(embedder, clf, dwell, "lets discuss ai")
    label, state = run_turn(embedder, clf, dwell, "lets switch gears")

    assert label == TOPIC_RESET
    assert state == DwellState.POST_RESET


def test_post_reset_requires_new_intent(
    dwell_setup: tuple[Embedder, Classifier, LabelDwell],
) -> None:
    """Validates reset blocks intent until new REQUEST."""
    embedder, clf, dwell = dwell_setup
    run_turn(embedder, clf, dwell, "lets discuss ai")
    run_turn(embedder, clf, dwell, "lets switch gears")
    label, state = run_turn(embedder, clf, dwell, "cool")
    assert label == SILENT
    assert state == DwellState.POST_RESET
    label, state = run_turn(embedder, clf, dwell, "lets discuss robotics")

    assert label == REQUEST
    assert state == DwellState.INTENT
