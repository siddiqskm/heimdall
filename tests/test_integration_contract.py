"""
Integration contract test: validates heimdall when used as an embedded library.

Uses only the public API (import from heimdall). Runs the standard pipeline
(init → embed → classify → dwell → decide → route) and asserts:
- Chat directory and files (delta.json, prototypes.json, dwell.json) are created.
- Returned action and label are valid constants.

Run after "pip install ." or "poetry install" to validate the installed package.

Cleanup: No teardown is required. pytest's tmp_path fixture provides a temporary
directory that is automatically removed after each test. We do not mutate global
heimdall state (each test creates fresh config/embedder/classifier/dwell).
"""

import json
import logging
from pathlib import Path


# Public API only
def _import_heimdall():
    import heimdall as h
    return h


def _run_one_turn(h, config, embedder, conversation_id: str, message: str):
    """One turn: get or create classifier + dwell, run pipeline, return (action, label)."""
    clf = h.Classifier(config=config, chat_id=conversation_id)
    dwell = h.LabelDwell(config=config, chat_id=clf.chat_id)
    vec = embedder.encode(message)
    pred = clf.predict(vec, text=message)
    dwell_label = dwell.apply(pred.label, pred.activation)
    final_label = h.decide(
        dwell_label,
        pred.confidence,
        confidence_threshold=config.confidence_threshold,
    )
    action = h.route(final_label)
    return action, final_label


def test_integration_contract_files_and_returns(tmp_path: Path) -> None:
    """
    Contract: init with state_dir, run pipeline for one conversation; chat dir
    must contain delta.json, prototypes.json, dwell.json; action/label must be
    valid constants.
    """
    h = _import_heimdall()
    state_dir = tmp_path / "heimdall_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    config = h.HeimdallConfig(state_dir=state_dir)
    h.set_log_level(logging.INFO)
    embedder = h.Embedder()

    conversation_id = "contract_conv_1"
    action, label = _run_one_turn(h, config, embedder, conversation_id, "lets discuss auth")

    # ---- Valid actions and labels (public constants) ----
    valid_actions = (h.ALLOW_PROGRESS, h.NO_RESPONSE, h.RESET_CONTEXT, h.SUPPRESS)
    valid_labels = (h.SILENT, h.REQUEST, h.TOPIC_RESET, h.HOSTILE)

    assert action in valid_actions, f"action {action!r} not in {valid_actions}"
    assert label in valid_labels, f"label {label!r} not in {valid_labels}"

    # ---- Chat dir and persisted files ----
    chat_dir = config.chat_dir(conversation_id)
    assert chat_dir.exists() and chat_dir.is_dir()

    delta_path = chat_dir / "delta.json"
    proto_path = chat_dir / "prototypes.json"
    dwell_path = chat_dir / "dwell.json"

    assert delta_path.exists(), "delta.json should exist after one turn"
    assert proto_path.exists(), "prototypes.json should exist after one turn"
    assert dwell_path.exists(), "dwell.json should exist after one turn"

    with delta_path.open() as f:
        delta_data = json.load(f)
    assert isinstance(delta_data, list), "delta.json should be a list (bias vector)"

    with proto_path.open() as f:
        proto_data = json.load(f)
    assert isinstance(proto_data, dict), "prototypes.json should be a dict"

    with dwell_path.open() as f:
        dwell_data = json.load(f)
    assert isinstance(dwell_data, dict) and "state" in dwell_data, "dwell.json should have 'state'"


def test_integration_contract_second_turn_same_conversation(tmp_path: Path) -> None:
    """
    Contract: two turns with the same conversation_id reuse the same chat dir
    and update the persisted files.
    """
    h = _import_heimdall()
    state_dir = tmp_path / "heimdall_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    config = h.HeimdallConfig(state_dir=state_dir)
    h.set_log_level(logging.INFO)
    embedder = h.Embedder()

    conversation_id = "contract_conv_2"
    _run_one_turn(h, config, embedder, conversation_id, "hello")
    action2, label2 = _run_one_turn(h, config, embedder, conversation_id, "lets talk about api design")

    valid_actions = (h.ALLOW_PROGRESS, h.NO_RESPONSE, h.RESET_CONTEXT, h.SUPPRESS)
    valid_labels = (h.SILENT, h.REQUEST, h.TOPIC_RESET, h.HOSTILE)
    assert action2 in valid_actions
    assert label2 in valid_labels

    chat_dir = config.chat_dir(conversation_id)
    assert (chat_dir / "delta.json").exists()
    assert (chat_dir / "prototypes.json").exists()
    assert (chat_dir / "dwell.json").exists()
