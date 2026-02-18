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
) -> tuple[Embedder, Classifier, LabelDwell, str]:
    """
    Creates fresh Heimdall pipeline for FSM validation.
    """

    config: HeimdallConfig = HeimdallConfig(
        state_dir=tmp_path / "heimdall_state"
    )

    embedder: Embedder = Embedder()
    clf: Classifier = Classifier(config=config)
    dwell: LabelDwell = LabelDwell(config=config, debug=False)

    user_id: str = "fsm_test_user"

    return embedder, clf, dwell, user_id


def run_turn(
    embedder: Embedder,
    clf: Classifier,
    dwell: LabelDwell,
    user_id: str,
    text: str,
) -> tuple[Label, DwellState]:
    """
    Helper to execute a single turn and return final label and FSM state.
    """

    vector = embedder.encode(text)

    predicted: Label
    confidence: float
    activation: float

    predicted, confidence, activation = clf.predict(
        vector,
        user_id,
    )

    dwell_label: Label = dwell.apply(
        user_id,
        predicted,
        activation,
    )

    final_label: Label = decide(
        dwell_label,
        confidence,
    )

    state: DwellState = dwell.state[user_id]

    return final_label, state


def test_idle_to_intent_transition(
    dwell_setup: tuple[Embedder, Classifier, LabelDwell, str],
) -> None:
    """
    Validates IDLE → INTENT transition.
    """

    embedder, clf, dwell, user_id = dwell_setup

    label, state = run_turn(
        embedder,
        clf,
        dwell,
        user_id,
        "lets discuss space",
    )

    assert label == REQUEST
    assert state == DwellState.INTENT


def test_intent_continuity_with_acknowledgement(
    dwell_setup: tuple[Embedder, Classifier, LabelDwell, str],
) -> None:
    """
    Validates benign SILENT inherits intent.
    """

    embedder, clf, dwell, user_id = dwell_setup

    run_turn(embedder, clf, dwell, user_id, "lets discuss space")

    label, state = run_turn(
        embedder,
        clf,
        dwell,
        user_id,
        "cool",
    )

    assert label == REQUEST
    assert state == DwellState.INTENT


def test_intent_decay_on_second_silent(
    dwell_setup: tuple[Embedder, Classifier, LabelDwell, str],
) -> None:
    """
    Validates intent exits after decay threshold.
    """

    embedder, clf, dwell, user_id = dwell_setup

    run_turn(embedder, clf, dwell, user_id, "lets discuss space")

    run_turn(embedder, clf, dwell, user_id, "cool")

    label, state = run_turn(
        embedder,
        clf,
        dwell,
        user_id,
        "hey",
    )

    assert label == REQUEST
    assert state == DwellState.INTENT


def test_hostile_entry_and_recovery(
    dwell_setup: tuple[Embedder, Classifier, LabelDwell, str],
) -> None:
    """
    Validates HOSTILE entry and cooldown-based recovery.
    """

    embedder, clf, dwell, user_id = dwell_setup

    # enter hostile
    label, state = run_turn(
        embedder,
        clf,
        dwell,
        user_id,
        "what the hell",
    )

    assert label == HOSTILE
    assert state == DwellState.HOSTILE

    # cooldown turn 1
    label, state = run_turn(
        embedder,
        clf,
        dwell,
        user_id,
        "lets continue",
    )

    assert label == HOSTILE
    assert state == DwellState.HOSTILE

    # cooldown turn 2
    label, state = run_turn(
        embedder,
        clf,
        dwell,
        user_id,
        "continue please",
    )

    assert label == HOSTILE
    assert state == DwellState.HOSTILE

    # recovery allowed now
    label, state = run_turn(
        embedder,
        clf,
        dwell,
        user_id,
        "lets continue",
    )

    assert label == REQUEST
    assert state == DwellState.INTENT


def test_topic_reset_transition(
    dwell_setup: tuple[Embedder, Classifier, LabelDwell, str],
) -> None:
    """
    Validates reset forces POST_RESET state.
    """

    embedder, clf, dwell, user_id = dwell_setup

    run_turn(embedder, clf, dwell, user_id, "lets discuss ai")

    label, state = run_turn(
        embedder,
        clf,
        dwell,
        user_id,
        "lets switch gears",
    )

    assert label == TOPIC_RESET
    assert state == DwellState.POST_RESET


def test_post_reset_requires_new_intent(
    dwell_setup: tuple[Embedder, Classifier, LabelDwell, str],
) -> None:
    """
    Validates reset blocks intent until new REQUEST.
    """

    embedder, clf, dwell, user_id = dwell_setup

    run_turn(embedder, clf, dwell, user_id, "lets discuss ai")

    run_turn(embedder, clf, dwell, user_id, "lets switch gears")

    label, state = run_turn(
        embedder,
        clf,
        dwell,
        user_id,
        "cool",
    )

    assert label == SILENT
    assert state == DwellState.POST_RESET

    label, state = run_turn(
        embedder,
        clf,
        dwell,
        user_id,
        "lets discuss robotics",
    )

    assert label == REQUEST
    assert state == DwellState.INTENT
