# tests/conftest.py

import logging

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--log-heimdall",
        action="store",
        default=None,
        metavar="LEVEL",
        help="Enable heimdall package logging during tests (e.g. INFO, DEBUG).",
    )


def pytest_configure(config: pytest.Config) -> None:
    level_name = config.getoption("--log-heimdall", None)
    if level_name is None:
        return

    level = getattr(logging, level_name.upper(), None)
    if level is None:
        try:
            level = int(level_name)
        except ValueError:
            logging.warning(
                "Unknown log level %r for --log-heimdall; use e.g. INFO or DEBUG",
                level_name,
            )
            return

    import heimdall

    heimdall.configure_logging(level=level)
