# heimdall/core/classifier.py

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from heimdall.adapt.config import DECAY, MAX_BIAS
from heimdall.core.config import HeimdallConfig, default_config
from heimdall.core.model_loader import load_lr_model, load_offline_prototypes
from heimdall.core.persistence import load_user_delta, save_user_delta
from heimdall.core.persistence_prototypes import load_prototypes, save_prototypes
from heimdall.core.prototypes import PrototypeStore
from heimdall.core.types import (
    ID_TO_LABEL,
    LABELS,
    REQUEST,
    SILENT,
    STEER,
    Label,
)

type UserID = str
type EmbeddingVector = NDArray[np.float64]
type BiasVector = NDArray[np.float64]


class Classifier:
    def __init__(
        self,
        config: HeimdallConfig | None = None,
    ) -> None:
        # ------------------------------------------------------------------
        # Configuration
        # ------------------------------------------------------------------
        self.config = config or default_config()
        self.config.ensure_dirs()

        # ------------------------------------------------------------------
        # Model (immutable package asset)
        # ------------------------------------------------------------------
        self.model = load_lr_model()
        self._normalize_model()

        # ------------------------------------------------------------------
        # Runtime persistence paths (user-scoped state only)
        # ------------------------------------------------------------------
        self.state_path: Path = self.config.state_dir / "user_delta.json"
        self.proto_user_path: Path = (
            self.config.state_dir / "prototypes_user.json"
        )

        # ------------------------------------------------------------------
        # Bias state
        # ------------------------------------------------------------------
        self.user_delta: dict[UserID, BiasVector] = load_user_delta(
            self.state_path
        )
        self.num_labels: int = len(LABELS)

        # ------------------------------------------------------------------
        # Prototype tiers
        # ------------------------------------------------------------------

        # Session prototypes (ephemeral, cleared per session)
        self.session_prototypes = PrototypeStore(
            max_per_label=self.config.session_proto_limit
        )

        # User prototypes (persisted per user)
        self.user_prototypes = PrototypeStore(
            max_per_label=self.config.user_proto_limit
        )
        self.user_prototypes.store = load_prototypes(
            self.proto_user_path
        )

        # Offline prototypes (immutable, package-scoped)
        # These are NOT runtime state and must never live in state_dir.
        self.offline_prototypes = PrototypeStore(
            max_per_label=self.config.offline_proto_limit
        )
        self.offline_prototypes.store = load_offline_prototypes()

    # ------------------------------------------------------------------
    # Model compatibility hardening
    # ------------------------------------------------------------------
    def _normalize_model(self) -> None:
        """
        Normalize sklearn LogisticRegression models that were pickled
        under a newer sklearn version.

        Avoids runtime crashes such as:
        AttributeError: 'LogisticRegression' object has no attribute 'multi_class'
        """

        if not hasattr(self.model, "predict_proba"):
            raise TypeError(
                "Loaded model does not implement predict_proba()"
            )

        # Compatibility shim for cross-version sklearn pickles
        if not hasattr(self.model, "multi_class"):
            try:
                n_classes = len(self.model.classes_)
            except Exception:
                n_classes = 2

            self.model.multi_class = (
                "multinomial" if n_classes > 2 else "auto"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_user(self, user_id: UserID) -> None:
        if user_id not in self.user_delta:
            self.user_delta[user_id] = np.zeros(self.num_labels)

    def _apply_decay(self, user_id: UserID) -> None:
        self.user_delta[user_id] *= DECAY
        self.user_delta[user_id][
            np.abs(self.user_delta[user_id]) < 1e-4
        ] = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(
        self,
        vector: EmbeddingVector,
        user_id: UserID,
    ) -> tuple[Label, float, float]:
        """
        Predict intent label.

        Returns:
            (label, confidence, activation)

        Activation is used by dwell.
        Confidence is used by decision gate.
        """

        # --------------------------------------------------------------
        # SESSION PROTOTYPES (highest priority, always allowed)
        # --------------------------------------------------------------
        proto_label, activation = self.session_prototypes.match(
            vector,
            threshold=self.config.session_proto_threshold,
        )
        if proto_label is not None:
            return proto_label, activation, activation

        # --------------------------------------------------------------
        # USER PROTOTYPES (soft prior)
        # --------------------------------------------------------------
        proto_label, activation = self.user_prototypes.match(
            vector,
            threshold=self.config.user_proto_threshold,
        )
        if proto_label is not None:
            return proto_label, activation, activation

        # --------------------------------------------------------------
        # LR FALLBACK (core statistical intent)
        # --------------------------------------------------------------
        probs: NDArray[np.float64] = (
            self.model.predict_proba([vector])[0]
        )

        self._init_user(user_id)
        self._apply_decay(user_id)

        biased_probs = probs + self.user_delta[user_id]
        biased_probs = np.clip(biased_probs, 1e-6, 1.0)
        biased_probs = biased_probs / biased_probs.sum()

        idx: int = int(np.argmax(biased_probs))
        label: Label = ID_TO_LABEL[idx]
        confidence: float = float(biased_probs[idx])

        # --------------------------------------------------------------
        # OFFLINE PROTOTYPES (guarded override)
        # --------------------------------------------------------------
        proto_label, activation = self.offline_prototypes.match(
            vector,
            threshold=self.config.offline_proto_threshold,
        )

        # IMPORTANT RULE:
        # Do NOT allow STEER/SILENT to override an active REQUEST
        if (
            proto_label is not None
            and not (
                label == REQUEST
                and proto_label in {STEER, SILENT}
            )
        ):
            return proto_label, activation, activation

        return label, confidence, confidence

    def maybe_add_prototype(
        self,
        label: Label,
        vector: EmbeddingVector,
        confidence: float,
    ) -> None:
        """
        Add prototypes based on confidence thresholds.
        """

        if label == SILENT:
            return

        if confidence >= 0.55:
            self.session_prototypes.add(label, vector)

        if confidence >= 0.65:
            self.user_prototypes.add(label, vector)

    def update_bias(
        self,
        user_id: UserID,
        label_index: int,
        delta: float,
    ) -> None:
        """
        Adjust per-user bias vector.
        """

        self._init_user(user_id)
        self.user_delta[user_id][label_index] = np.clip(
            self.user_delta[user_id][label_index] + delta,
            -MAX_BIAS,
            MAX_BIAS,
        )

    def persist(self) -> None:
        """
        Persist user-scoped runtime state.
        """

        save_user_delta(self.state_path, self.user_delta)
        save_prototypes(
            self.proto_user_path,
            self.user_prototypes.store,
        )

    def end_session(self) -> None:
        """
        Clear session-scoped prototypes.
        """
        self.session_prototypes.clear()

    def reset_user(self, user_id: str) -> None:
        """
        Hard reset user bias (used in tests).
        """
        self.user_delta.pop(user_id, None)
