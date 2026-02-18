# tests/conftest.py

import logging
from pathlib import Path

import pytest

# Project root (parent of tests/); test logs go to .heimdall/ in project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _PROJECT_ROOT / ".heimdall"
_LOG_FILE = _LOG_DIR / "heimdall.log"

# Flush after each emit so logs appear in file immediately (pytest can buffer otherwise)
class _FlushingFileHandler(logging.FileHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--log-heimdall",
        action="store",
        default="INFO",
        metavar="LEVEL",
        help="Heimdall and tests log level (default: INFO). Use 0 or NONE to disable.",
    )


def _resolve_level(level_name: str) -> int | None:
    if not level_name or str(level_name).upper() in ("0", "NONE", "OFF"):
        return None
    level = getattr(logging, str(level_name).upper(), None)
    if level is not None:
        return level
    try:
        return int(level_name)
    except (ValueError, TypeError):
        return None


def _configure_heimdall_logging(config: pytest.Config) -> None:
    level_name = config.getoption("--log-heimdall", "INFO")
    level = _resolve_level(level_name)
    if level is None:
        if level_name and str(level_name).upper() not in ("0", "NONE", "OFF"):
            logging.warning(
                "Unknown log level %r for --log-heimdall; use e.g. INFO, DEBUG, or 0 to disable",
                level_name,
            )
        return

    import heimdall

    heimdall.configure_logging(
        level=level,
        log_file=False,  # we add file handler on root below so logs reliably land in file
        log_dir=_LOG_DIR,
        also_configure=["tests"],
    )

    # Ensure logs land in file: attach file handler to root logger and make heimdall
    # propagate so all heimdall.* (and tests.*) logs reach the file. Pytest's capture
    # or process boundaries can prevent logs from reaching only the heimdall logger.
    root = logging.getLogger()
    root.setLevel(min(root.level, level))
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_h = _FlushingFileHandler(_LOG_FILE, encoding="utf-8")
    file_h.setLevel(level)
    file_h.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root.addHandler(file_h)
    logging.getLogger("heimdall").propagate = True
    for name in ("tests",):
        logging.getLogger(name).propagate = True


def pytest_configure(config: pytest.Config) -> None:
    _configure_heimdall_logging(config)


@pytest.fixture(scope="session", autouse=True)
def _heimdall_logging_session(request: pytest.FixtureRequest) -> None:
    """Configure heimdall logging in the process that runs tests (fixes empty log file with xdist or late import)."""
    _configure_heimdall_logging(request.config)
