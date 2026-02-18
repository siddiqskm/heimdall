# tests/test_logging_config.py
"""Suite 4: Logging – default under .heimdall, custom log_dir, explicit path, no file when disabled."""

import logging
from pathlib import Path

import pytest

import heimdall


def test_default_log_under_heimdall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """4.1: log_file=True with default log_dir creates ~/.heimdall/heimdall.log."""
    monkeypatch.setattr(Path, "home", lambda *args, **kwargs: tmp_path)
    heimdall.configure_logging(log_file=True, use_console=False)
    log_path = tmp_path / ".heimdall" / "heimdall.log"
    assert log_path.parent.exists()
    logging.getLogger("heimdall").info("default_log_test")
    for h in logging.getLogger("heimdall").handlers:
        h.flush()
    assert log_path.exists()
    assert "default_log_test" in log_path.read_text()


def test_custom_log_dir_when_log_file_true(tmp_path: Path) -> None:
    """4.2: log_file=True and log_dir=custom creates custom/heimdall.log."""
    log_dir = tmp_path / "logs"
    heimdall.configure_logging(log_file=True, log_dir=log_dir, use_console=False)
    log_path = log_dir / "heimdall.log"
    logging.getLogger("heimdall").info("custom_dir_test")
    for h in logging.getLogger("heimdall").handlers:
        h.flush()
    assert log_path.exists()
    assert "custom_dir_test" in log_path.read_text()


def test_explicit_log_path(tmp_path: Path) -> None:
    """4.3: log_file=path creates file at that path; parent dir created."""
    log_path = tmp_path / "my_logs" / "app.log"
    heimdall.configure_logging(log_file=log_path, use_console=False)
    logging.getLogger("heimdall").info("explicit_path_test")
    for h in logging.getLogger("heimdall").handlers:
        h.flush()
    assert log_path.exists()
    assert "explicit_path_test" in log_path.read_text()


def test_log_file_false_adds_no_file_handler(tmp_path: Path) -> None:
    """4.4: log_file=False or None adds no FileHandler; no file under .heimdall."""
    heimdall.configure_logging(log_file=False, use_console=True)
    log = logging.getLogger("heimdall")
    file_handlers = [h for h in log.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 0

    default_log = tmp_path / ".heimdall" / "heimdall.log"
    assert not default_log.exists()
