# tests/test_default_config_state_dir.py

import json
from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch

from heimdall.core.classifier import Classifier
from heimdall.core.config import default_config


def test_default_config_creates_state_and_files(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """
    Ensure that:

    1. default_config() resolves to ~/.heimdall
    2. Classifier creates the directory
    3. persist() creates canonical runtime JSON files
    4. Files contain valid JSON
    """

    # Redirect HOME to temporary directory
    monkeypatch.setenv("HOME", str(tmp_path))

    # Instantiate classifier using default config
    clf = Classifier()
    cfg = default_config()

    expected_dir = tmp_path / ".heimdall"

    # ---- Directory must exist ----
    assert cfg.state_dir == expected_dir
    assert expected_dir.exists()
    assert expected_dir.is_dir()

    # ---- Persist runtime state (per-chat) ----
    clf.persist()

    # ---- Chat-scoped paths ----
    chat_dir = cfg.chat_dir(clf.chat_id)
    delta_file = chat_dir / "delta.json"
    proto_file = chat_dir / "prototypes.json"

    assert delta_file.exists()
    assert proto_file.exists()

    # ---- Delta is a single bias vector (list); prototypes are dict ----
    with delta_file.open() as f:
        delta_data = json.load(f)
        assert isinstance(delta_data, list)
    with proto_file.open() as f:
        proto_data = json.load(f)
        assert isinstance(proto_data, dict)
