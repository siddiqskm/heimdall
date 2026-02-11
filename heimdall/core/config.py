# heimdall/core/config.py

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HeimdallConfig:
    """
    Runtime configuration for Heimdall.

    Only `state_dir` is configurable.

    All runtime filenames are canonical and enforced.
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

    # ------------------------------------------------------------------
    # Canonical runtime paths (NOT configurable)
    # ------------------------------------------------------------------

    @property
    def user_delta_path(self) -> Path:
        return self.state_dir / "user_delta.json"

    @property
    def user_prototypes_path(self) -> Path:
        return self.state_dir / "prototypes_user.json"

    # ------------------------------------------------------------------

    def ensure_dirs(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)


def default_config() -> HeimdallConfig:
    home = Path.home()
    return HeimdallConfig(
        state_dir=home / ".heimdall"
    )
