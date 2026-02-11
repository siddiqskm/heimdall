# heimdall/core/config.py

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HeimdallConfig:
    """
    Runtime configuration for Heimdall.

    state_dir:
        Directory where runtime JSON state is stored.

    models_dir:
        Optional override for model storage.

    --- Behavioral parameters ---
    session_proto_threshold:
        Threshold for session prototype activation.

    user_proto_threshold:
        Threshold for user prototype activation.

    offline_proto_threshold:
        Threshold for offline prototype activation.

    session_proto_limit:
        Max session prototypes per label.

    user_proto_limit:
        Max user prototypes per label.

    offline_proto_limit:
        Max offline prototypes per label.
    """

    state_dir: Path
    models_dir: Path | None = None

    # ---- thresholds ----
    session_proto_threshold: float = 0.75
    user_proto_threshold: float = 0.80
    offline_proto_threshold: float = 0.85

    # ---- capacity limits ----
    session_proto_limit: int = 5
    user_proto_limit: int = 15
    offline_proto_limit: int = 50

    def ensure_dirs(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)


def default_config() -> HeimdallConfig:
    home = Path.home()
    return HeimdallConfig(
        state_dir=home / ".heimdall"
    )
