# heimdall/__init__.py

import logging

# Prevent "no handler" warnings when the package is used without configuring logging.
# Applications that want output should call configure_logging() once.
_logger = logging.getLogger("heimdall")
if not _logger.handlers:
    _logger.addHandler(logging.NullHandler())


def configure_logging(
    level: int = logging.INFO,
    format_string: str = "%(message)s",
    *,
    handler: logging.Handler | None = None,
) -> None:
    """
    Configure logging for the heimdall package.

    Call this once from your application (or script) if you want heimdall
    log messages to be emitted. If not called, the package uses a NullHandler
    and no library logs are output.

    Args:
        level: Log level for the heimdall logger (default INFO).
        format_string: Format for log records (default "%(message)s").
        handler: Optional handler to add. If None, a StreamHandler(stderr) is used.
    """
    _logger.setLevel(level)
    _logger.propagate = False

    # Remove existing non-Null handlers so we don't duplicate
    for h in _logger.handlers[:]:
        if not isinstance(h, logging.NullHandler):
            _logger.removeHandler(h)

    if handler is None:
        # Default: emit to stderr (no log file is created).
        handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(format_string))
    _logger.addHandler(handler)
