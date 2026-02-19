# heimdall/core/config.py

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HeimdallConfig:
    """
    Runtime configuration for Heimdall.

    Only `state_dir` is required. Per-chat state is stored under state_dir/chats/{chat_id}/.
    Default is ~/.heimdall (use default_config()); pass any path to override (e.g. .playground_state).
    Assets and models are package-scoped by default, but may be overridden.
    """

    state_dir: Path
    models_dir: Path | None = None
    assets_dir: Path | None = None

    # ---- prototype thresholds ----
    session_proto_threshold: float = 0.75
    user_proto_threshold: float = 0.80
    offline_proto_threshold: float = 0.80

    # ---- prototype limits ----
    session_proto_limit: int = 5
    user_proto_limit: int = 15
    offline_proto_limit: int = 50

    # ---- prototype add thresholds (when to add to session/user store) ----
    session_proto_add_threshold: float = 0.55
    user_proto_add_threshold: float = 0.65

    # ------------------------------------------------------------------
    # Decision gate (SILENT → REQUEST when confidence below this)
    # ------------------------------------------------------------------

    confidence_threshold: float = 0.38

    # ------------------------------------------------------------------
    # Dwell FSM
    # ------------------------------------------------------------------

    hostile_cooldown: int = 2
    intent_decay_silent_streak: int = 2
    hostile_recovery_threshold: float = 0.5

    # ------------------------------------------------------------------
    # Score-based thresholds
    # ------------------------------------------------------------------

    hostile_threshold: float = 0.80
    reset_threshold: float = 0.65
    utility_silent_threshold: float = 0.40

    # ---- soft nearest-neighbor fallback (when LR confidence is low) ----
    lr_low_confidence_threshold: float = 0.45
    soft_proto_threshold: float = 0.65

    # ------------------------------------------------------------------
    # Score weights
    # ------------------------------------------------------------------

    drift_weight: float = 0.70

    novelty_weight: float = 0.50
    info_density_weight: float = 0.30
    richness_weight: float = 0.20

    # ------------------------------------------------------------------
    # Learning gate (when to allow learning side-effects)
    # ------------------------------------------------------------------

    learning_gate_min_confidence: float = 0.35
    learning_gate_min_stable_turns: int = 2
    learning_gate_min_interval_sec: float = 30.0

    # ------------------------------------------------------------------
    # Outcome inference (adapt)
    # ------------------------------------------------------------------

    outcome_escalated_confidence_threshold: float = 0.4
    outcome_escalated_min_next_len: int = 20

    # ------------------------------------------------------------------
    # Runtime state paths (chat-scoped)
    # ------------------------------------------------------------------

    def chat_dir(self, chat_id: str) -> Path:
        """Directory for persisting state for one chat."""
        return self.state_dir / "chats" / chat_id

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

    def ensure_chat_dir(self, chat_id: str) -> None:
        """Ensure directory for a chat exists."""
        self.chat_dir(chat_id).mkdir(parents=True, exist_ok=True)

    def delete_chat_state(self, chat_id: str) -> None:
        """Remove persisted state for this chat. Idempotent if dir already missing."""
        path = self.chat_dir(chat_id)
        if path.exists():
            shutil.rmtree(path)


def default_config() -> HeimdallConfig:
    home = Path.home()
    return HeimdallConfig(
        state_dir=home / ".heimdall"
    )
