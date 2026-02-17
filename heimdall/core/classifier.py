# heimdall/core/classifier.py

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from heimdall.adapt.config import DECAY, MAX_BIAS
from heimdall.core.config import HeimdallConfig, default_config
from heimdall.core.model_loader import (
    load_lr_model,
    load_offline_prototypes,
)
from heimdall.core.persistence import load_user_delta, save_user_delta
from heimdall.core.persistence_prototypes import (
    load_prototypes,
    save_prototypes,
)
from heimdall.core.prototypes import PrototypeStore
from heimdall.core.score_engine import ScoreEngine, Scores
from heimdall.core.types import (
    HOSTILE,
    ID_TO_LABEL,
    LABELS,
    REQUEST,
    SILENT,
    TOPIC_RESET,
    Label,
)

type UserID = str
type EmbeddingVector = NDArray[np.float64]
type BiasVector = NDArray[np.float64]


# ==========================================================
# Prediction Object (Backward Compatible)
# ==========================================================

@dataclass(frozen=True)
class Prediction:
    label: Label
    confidence: float
    activation: float
    hostile_score: float
    reset_score: float
    utility_score: float

    # backward compatibility: allow tuple unpacking
    def __iter__(self) -> Iterator[Any]:
        yield self.label
        yield self.confidence
        yield self.activation


# ==========================================================
# Classifier
# ==========================================================

class Classifier:
    def __init__(
        self,
        config: HeimdallConfig | None = None,
    ) -> None:

        self.config = config or default_config()
        self.config.ensure_dirs()

        # ---- Model ----
        self.model = load_lr_model(self.config.lr_model_path)
        self._normalize_model()

        # ---- Paths ----
        self.state_path: Path = self.config.user_delta_path
        self.proto_user_path: Path = self.config.user_prototypes_path

        # ---- Bias ----
        self.user_delta: dict[UserID, BiasVector] = load_user_delta(
            self.state_path
        )
        self.num_labels: int = len(LABELS)

        # ---- Prototypes ----
        self.session_prototypes = PrototypeStore(
            max_per_label=self.config.session_proto_limit
        )

        self.user_prototypes = PrototypeStore(
            max_per_label=self.config.user_proto_limit
        )
        self.user_prototypes.store = load_prototypes(
            self.proto_user_path
        )

        self.offline_prototypes = PrototypeStore(
            max_per_label=self.config.offline_proto_limit
        )
        self.offline_prototypes.store = load_offline_prototypes(
            self.config.offline_prototypes_path
        )

        # ---- Score Engine ----
        self._score_engine = ScoreEngine(self.config)

        # ---- Context memory ----
        self._recent_vectors: dict[UserID, list[EmbeddingVector]] = {}
        self._max_recent: int = 20

    # ------------------------------------------------------------------
    # Model compatibility
    # ------------------------------------------------------------------

    def _normalize_model(self) -> None:
        if not hasattr(self.model, "predict_proba"):
            raise TypeError(
                "Loaded model does not implement predict_proba()"
            )

        if not hasattr(self.model, "multi_class"):
            try:
                n_classes = len(self.model.classes_)
            except Exception:
                n_classes = 2

            self.model.multi_class = (
                "multinomial" if n_classes > 2 else "auto"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_user(self, user_id: UserID) -> None:
        if user_id not in self.user_delta:
            self.user_delta[user_id] = np.zeros(self.num_labels)

        if user_id not in self._recent_vectors:
            self._recent_vectors[user_id] = []

    def _apply_decay(self, user_id: UserID) -> None:
        self.user_delta[user_id] *= DECAY
        self.user_delta[user_id][
            np.abs(self.user_delta[user_id]) < 1e-4
        ] = 0.0

    def _update_context(
        self,
        user_id: UserID,
        vector: EmbeddingVector,
    ) -> None:

        self._recent_vectors[user_id].append(vector)

        if len(self._recent_vectors[user_id]) > self._max_recent:
            self._recent_vectors[user_id] = self._recent_vectors[user_id][
                -self._max_recent :
            ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        vector: EmbeddingVector,
        user_id: UserID,
        text: str | None = None,
    ) -> Prediction:

        self._init_user(user_id)

        # --------------------------------------------------------------
        # Compute Scores
        # --------------------------------------------------------------

        scores: Scores = self._score_engine.compute(
            vector=vector,
            text=text or "",
            recent_vectors=self._recent_vectors[user_id],
            offline_prototypes=self.offline_prototypes,
            user_prototypes=self.user_prototypes,
            session_prototypes=self.session_prototypes,
        )

        # --------------------------------------------------------------
        # ORIGINAL LR / PROTOTYPE LOGIC (Baseline)
        # --------------------------------------------------------------

        # --- Session prototypes ---
        proto_label, activation = self.session_prototypes.match(
            vector,
            threshold=self.config.session_proto_threshold,
        )
        if proto_label is not None:
            self._update_context(user_id, vector)
            return Prediction(
                proto_label,
                activation,
                activation,
                scores.hostile_score,
                scores.reset_score,
                scores.utility_score,
            )

        # --- User prototypes ---
        proto_label, activation = self.user_prototypes.match(
            vector,
            threshold=self.config.user_proto_threshold,
        )
        if proto_label is not None:
            self._update_context(user_id, vector)
            return Prediction(
                proto_label,
                activation,
                activation,
                scores.hostile_score,
                scores.reset_score,
                scores.utility_score,
            )

        # --- LR fallback ---
        probs: NDArray[np.float64] = (
            self.model.predict_proba([vector])[0]
        )

        self._apply_decay(user_id)

        biased_probs = probs + self.user_delta[user_id]
        biased_probs = np.clip(biased_probs, 1e-6, 1.0)
        biased_probs = biased_probs / biased_probs.sum()

        idx: int = int(np.argmax(biased_probs))
        lr_label: Label = ID_TO_LABEL[idx]
        lr_conf: float = float(biased_probs[idx])

        # --- Offline prototypes override ---
        proto_label, activation = self.offline_prototypes.match(
            vector,
            threshold=self.config.offline_proto_threshold,
        )

        if (
            proto_label is not None
            and not (
                lr_label == REQUEST
                and proto_label == SILENT
            )
        ):
            self._update_context(user_id, vector)
            return Prediction(
                proto_label,
                activation,
                activation,
                scores.hostile_score,
                scores.reset_score,
                scores.utility_score,
            )

        # --------------------------------------------------------------
        # SCORE-BASED LABEL DERIVATION (NEW LAYER)
        # --------------------------------------------------------------

        score_label: Label = lr_label
        confidence: float = lr_conf

        # 1. Hostile override (strongest)
        if scores.hostile_score >= self.config.hostile_threshold:
            score_label = HOSTILE
            confidence = scores.hostile_score

        # 2. Reset override (conservative)
        elif scores.reset_score >= self.config.reset_threshold:
            score_label = TOPIC_RESET
            confidence = scores.reset_score

        # --------------------------------------------------------------
        # Drift Logging (diagnostic only)
        # --------------------------------------------------------------

        if score_label != lr_label:
            print(
                f"[Heimdall Drift] "
                f"LR={lr_label} "
                f"SCORE={score_label} | "
                f"H={scores.hostile_score:.2f} "
                f"R={scores.reset_score:.2f} "
                f"U={scores.utility_score:.2f}"
            )

        # --------------------------------------------------------------
        # Update Context
        # --------------------------------------------------------------

        self._update_context(user_id, vector)

        # --------------------------------------------------------------
        # Final Prediction
        # --------------------------------------------------------------

        return Prediction(
            score_label,
            confidence,
            confidence,
            scores.hostile_score,
            scores.reset_score,
            scores.utility_score,
        )

    # ------------------------------------------------------------------
    # Existing methods unchanged
    # ------------------------------------------------------------------

    def maybe_add_prototype(
        self,
        label: Label,
        vector: EmbeddingVector,
        confidence: float,
    ) -> None:
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
        self._init_user(user_id)
        self.user_delta[user_id][label_index] = np.clip(
            self.user_delta[user_id][label_index] + delta,
            -MAX_BIAS,
            MAX_BIAS,
        )

    def persist(self) -> None:
        save_user_delta(self.state_path, self.user_delta)
        save_prototypes(
            self.proto_user_path,
            self.user_prototypes.store,
        )

    def end_session(self) -> None:
        self.session_prototypes.clear()

    def reset_user(self, user_id: str) -> None:
        self.user_delta.pop(user_id, None)
        self._recent_vectors.pop(user_id, None)
