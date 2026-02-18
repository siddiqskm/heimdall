# heimdall/__init__.py

import logging
from pathlib import Path

# Prevent "no handler" warnings when the package is used without configuring logging.
# Applications that want output should call configure_logging() once.
_logger = logging.getLogger("heimdall")
if not _logger.handlers:
    _logger.addHandler(logging.NullHandler())

# Default format for file handler (includes timestamp); console can stay minimal.
_DEFAULT_FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DEFAULT_LOG_FILENAME = "heimdall.log"


class _FlushingFileHandler(logging.FileHandler):
    """FileHandler that flushes after each emit so logs appear immediately."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def configure_logging(
    level: int = logging.INFO,
    format_string: str = "%(message)s",
    *,
    handler: logging.Handler | None = None,
    log_file: bool | str | Path | None = None,
    log_dir: str | Path | None = None,
    use_console: bool = True,
    also_configure: list[str] | None = None,
) -> None:
    """
    Configure logging for the heimdall package.

    Call this once from your application (or script) if you want heimdall
    log messages to be emitted. When running in the background (e.g. behind a
    chat API), prefer setting log_file so logs are written to a file.

    Args:
        level: Log level for the heimdall logger (default INFO).
        format_string: Format for log records (default "%(message)s").
            Used for console; log file uses a format that includes timestamp and level.
        handler: Optional handler to add. If None, a StreamHandler(stderr) is added
            when use_console is True.
        log_file: If True, logs are written to a file at default location
            (log_dir or ~/.heimdall)/heimdall.log. If a path (str or Path), that path
            is used. If None or False, no file handler is added.
        log_dir: When log_file is True, directory for the log file. Default is
            ~/.heimdall. Ignored when log_file is an explicit path.
        use_console: If True and handler is None, add a StreamHandler. Set False to
            emit only to log_file (when log_file is set).
        also_configure: Optional list of logger names to attach the same handlers to
            (e.g. ["tests"] so test modules using getLogger("tests...") emit to the same place).
    """
    _logger.setLevel(level)
    _logger.propagate = False

    for h in _logger.handlers[:]:
        if not isinstance(h, logging.NullHandler):
            _logger.removeHandler(h)

    handlers: list[logging.Handler] = []

    if handler is not None:
        handlers.append(handler)
    else:
        if use_console:
            stream = logging.StreamHandler()
            stream.setLevel(level)
            stream.setFormatter(logging.Formatter(format_string))
            handlers.append(stream)
        # Resolve log file path: True -> default dir + filename; path -> as given
        if log_file is True:
            base = Path(log_dir) if log_dir is not None else Path.home() / ".heimdall"
            path = base / _DEFAULT_LOG_FILENAME
        elif log_file is not None and log_file is not False:
            path = Path(log_file)
        else:
            path = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_h = _FlushingFileHandler(path, encoding="utf-8")
            file_h.setLevel(level)
            file_h.setFormatter(logging.Formatter(_DEFAULT_FILE_FORMAT))
            handlers.append(file_h)

    for h in handlers:
        h.setLevel(level)
        if h.formatter is None:
            h.setFormatter(logging.Formatter(format_string))
        _logger.addHandler(h)

    if also_configure:
        for name in also_configure:
            other = logging.getLogger(name)
            other.setLevel(level)
            for h in handlers:
                other.addHandler(h)
