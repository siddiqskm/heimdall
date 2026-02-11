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

    # ---- Persist runtime state ----
    clf.persist()

    # ---- Canonical runtime paths ----
    user_delta_file = cfg.user_delta_path
    user_proto_file = cfg.user_prototypes_path

    # ---- Files must exist ----
    assert user_delta_file.exists()
    assert user_proto_file.exists()

    # ---- Files must contain valid JSON ----
    with user_delta_file.open() as f:
        user_delta_data = json.load(f)
        assert isinstance(user_delta_data, dict)

    with user_proto_file.open() as f:
        user_proto_data = json.load(f)
        assert isinstance(user_proto_data, dict)
