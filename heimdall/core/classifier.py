# heimdall/core/classifier.py

import logging
import uuid
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
from heimdall.core.persistence import load_chat_delta, save_chat_delta
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

type EmbeddingVector = NDArray[np.float64]
type BiasVector = NDArray[np.float64]

logger = logging.getLogger(__name__)


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
    """
    One instance per chat. Pass chat_id to resume; omit to start a new chat.
    """

    def __init__(
        self,
        config: HeimdallConfig | None = None,
        chat_id: str | None = None,
    ) -> None:

        self.config = config or default_config()
        self.config.ensure_dirs()

        self._chat_id: str = chat_id if chat_id else uuid.uuid4().hex
        self.config.ensure_chat_dir(self._chat_id)

        # ---- Model ----
        self.model = load_lr_model(self.config.lr_model_path)
        self._normalize_model()

        # ---- Paths (per chat) ----
        chat_dir = self.config.chat_dir(self._chat_id)
        self._delta_path: Path = chat_dir / "delta.json"
        self._proto_path: Path = chat_dir / "prototypes.json"

        # ---- Bias (single vector for this chat) ----
        loaded = load_chat_delta(self._delta_path)
        self.num_labels: int = len(LABELS)
        self._bias: BiasVector = (
            loaded if loaded is not None else np.zeros(self.num_labels)
        )

        # ---- Prototypes ----
        self.session_prototypes = PrototypeStore(
            max_per_label=self.config.session_proto_limit
        )
        self.user_prototypes = PrototypeStore(
            max_per_label=self.config.user_proto_limit
        )
        self.user_prototypes.store = load_prototypes(self._proto_path)

        self.offline_prototypes = PrototypeStore(
            max_per_label=self.config.offline_proto_limit
        )
        self.offline_prototypes.store = load_offline_prototypes(
            self.config.offline_prototypes_path
        )

        # ---- Score Engine ----
        self._score_engine = ScoreEngine(self.config)

        # ---- Context memory (this chat only) ----
        self._recent_vectors: list[EmbeddingVector] = []
        self._max_recent: int = 20

    @property
    def chat_id(self) -> str:
        return self._chat_id

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

    def _apply_decay(self) -> None:
        self._bias *= DECAY
        self._bias[np.abs(self._bias) < 1e-4] = 0.0

    def _update_context(self, vector: EmbeddingVector) -> None:
        self._recent_vectors.append(vector)
        if len(self._recent_vectors) > self._max_recent:
            self._recent_vectors = self._recent_vectors[-self._max_recent :]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        vector: EmbeddingVector,
        text: str | None = None,
    ) -> Prediction:

        # --------------------------------------------------------------
        # Compute Scores
        # --------------------------------------------------------------

        scores: Scores = self._score_engine.compute(
            vector=vector,
            text=text or "",
            recent_vectors=self._recent_vectors,
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
            self._update_context(vector)
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
            self._update_context(vector)
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

        self._apply_decay()

        biased_probs = probs + self._bias
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
            self._update_context(vector)
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
            logger.debug(
                "Drift LR=%s SCORE=%s H=%.2f R=%.2f U=%.2f",
                lr_label,
                score_label,
                scores.hostile_score,
                scores.reset_score,
                scores.utility_score,
            )

        # --------------------------------------------------------------
        # Update Context
        # --------------------------------------------------------------

        self._update_context(vector)

        # Persist state so delta.json and prototypes.json exist (e.g. in Cog)
        self.persist()

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

        if confidence >= self.config.session_proto_add_threshold:
            self.session_prototypes.add(label, vector)

        if confidence >= self.config.user_proto_add_threshold:
            self.user_prototypes.add(label, vector)

    def update_bias(
        self,
        label_index: int,
        delta: float,
    ) -> None:
        self._bias[label_index] = np.clip(
            self._bias[label_index] + delta,
            -MAX_BIAS,
            MAX_BIAS,
        )

    def persist(self) -> None:
        save_chat_delta(self._delta_path, self._bias)
        save_prototypes(self._proto_path, self.user_prototypes.store)

    def end_session(self) -> None:
        self.session_prototypes.clear()

    def reset_chat(self) -> None:
        """Clear in-memory state for this chat and persist (bias zeroed, prototypes cleared)."""
        self._bias = np.zeros(self.num_labels)
        self._recent_vectors.clear()
        self.session_prototypes.clear()
        self.user_prototypes.clear()
        self.persist()
