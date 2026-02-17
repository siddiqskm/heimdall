# heimdall/core/config.py

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HeimdallConfig:
    """
    Runtime configuration for Heimdall.

    Only `state_dir` is required.
    Assets and models are package-scoped by default,
    but may be overridden.
    """

    state_dir: Path
    models_dir: Path | None = None
    assets_dir: Path | None = None

    # ---- prototype thresholds ----
    session_proto_threshold: float = 0.75
    user_proto_threshold: float = 0.80
    offline_proto_threshold: float = 0.85

    # ---- prototype limits ----
    session_proto_limit: int = 5
    user_proto_limit: int = 15
    offline_proto_limit: int = 50

    # ------------------------------------------------------------------
    # Score-based thresholds
    # ------------------------------------------------------------------

    hostile_threshold: float = 0.80
    reset_threshold: float = 0.75
    utility_silent_threshold: float = 0.40

    # ------------------------------------------------------------------
    # Score weights
    # ------------------------------------------------------------------

    drift_weight: float = 0.70

    novelty_weight: float = 0.50
    info_density_weight: float = 0.30
    richness_weight: float = 0.20

    # ------------------------------------------------------------------
    # Runtime state paths (user-scoped)
    # ------------------------------------------------------------------

    @property
    def user_delta_path(self) -> Path:
        return self.state_dir / "user_delta.json"

    @property
    def user_prototypes_path(self) -> Path:
        return self.state_dir / "prototypes_user.json"

    # ------------------------------------------------------------------
    # Package asset paths (immutable)
    # ------------------------------------------------------------------

    @property
    def resolved_assets_dir(self) -> Path:
        if self.assets_dir is not None:
            return self.assets_dir

        return Path(__file__).resolve().parent.parent / "assets"

    @property
    def offline_prototypes_path(self) -> Path:
        return self.resolved_assets_dir / "prototypes_offline.json"

    # ------------------------------------------------------------------
    # Model paths
    # ------------------------------------------------------------------

    @property
    def resolved_models_dir(self) -> Path:
        if self.models_dir is not None:
            return self.models_dir

        return Path(__file__).resolve().parent.parent / "models"

    @property
    def lr_model_path(self) -> Path:
        return self.resolved_models_dir / "lr.joblib"

    # ------------------------------------------------------------------

    def ensure_dirs(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)


def default_config() -> HeimdallConfig:
    home = Path.home()
    return HeimdallConfig(
        state_dir=home / ".heimdall"
    )
