# core/classifier.py

from typing import Dict, Tuple, TypeAlias
import joblib
import numpy as np
from numpy.typing import NDArray

from core.types import (
    ID_TO_LABEL,
    LABELS,
    Label,
    SILENT,
    STEER,
    REQUEST,
)
from core.persistence import load_user_delta, save_user_delta
from core.prototypes import PrototypeStore
from core.persistence_prototypes import load_prototypes, save_prototypes


UserID: TypeAlias = str
EmbeddingVector: TypeAlias = NDArray[np.float64]
BiasVector: TypeAlias = NDArray[np.float64]

MAX_BIAS: float = 0.25
DECAY: float = 0.98


class Classifier:
    def __init__(
        self,
        model_path: str,
        state_path: str = "state/user_delta.json",
        proto_user_path: str = "state/prototypes_user.json",
        proto_offline_path: str = "state/prototypes_offline.json",
    ) -> None:
        self.model = joblib.load(model_path)

        self.state_path = state_path
        self.proto_user_path = proto_user_path

        # ---- bias ----
        self.user_delta: Dict[UserID, BiasVector] = load_user_delta(state_path)
        self.num_labels: int = len(LABELS)

        # ---- prototype tiers ----
        self.session_prototypes = PrototypeStore(max_per_label=5)

        self.user_prototypes = PrototypeStore(max_per_label=15)
        self.user_prototypes.store = load_prototypes(proto_user_path)

        self.offline_prototypes = PrototypeStore(max_per_label=50)
        self.offline_prototypes.store = load_prototypes(proto_offline_path)

    def _init_user(self, user_id: UserID) -> None:
        if user_id not in self.user_delta:
            self.user_delta[user_id] = np.zeros(self.num_labels)

    def _apply_decay(self, user_id: UserID) -> None:
        self.user_delta[user_id] *= DECAY
        self.user_delta[user_id][np.abs(self.user_delta[user_id]) < 1e-4] = 0.0

    def predict(
        self,
        vector: EmbeddingVector,
        user_id: UserID,
    ) -> Tuple[Label, float, float]:

        # ---- SESSION PROTOTYPES (always allowed) ----
        proto_label, activation = self.session_prototypes.match(
            vector, threshold=0.75
        )
        if proto_label is not None:
            print(f"[PROTO SESSION] {proto_label} {activation:.2f}")
            return proto_label, activation, activation

        # ---- USER PROTOTYPES (soft prior) ----
        proto_label, activation = self.user_prototypes.match(
            vector, threshold=0.80
        )
        if proto_label is not None:
            print(f"[PROTO USER] {proto_label} {activation:.2f}")
            return proto_label, activation, activation

        # ---- LR FALLBACK (core intent) ----
        probs: NDArray[np.float64] = self.model.predict_proba([vector])[0]

        self._init_user(user_id)
        self._apply_decay(user_id)

        biased_probs = probs + self.user_delta[user_id]
        biased_probs = np.clip(biased_probs, 1e-6, 1.0)
        biased_probs = biased_probs / biased_probs.sum()

        idx: int = int(np.argmax(biased_probs))
        label: Label = ID_TO_LABEL[idx]
        confidence: float = float(biased_probs[idx])

        # ---- OFFLINE PROTOTYPES (GUARDED) ----
        proto_label, activation = self.offline_prototypes.match(
            vector, threshold=0.85
        )

        # IMPORTANT RULE:
        # Do NOT allow STEER/SILENT to override an active REQUEST
        if proto_label is not None:
            if not (
                label == REQUEST
                and proto_label in {STEER, SILENT}
            ):
                print(f"[PROTO OFFLINE] {proto_label} {activation:.2f}")
                return proto_label, activation, activation

        return label, confidence, confidence

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
        save_prototypes(self.proto_user_path, self.user_prototypes.store)

    def end_session(self) -> None:
        self.session_prototypes.clear()

    def reset_user(self, user_id: str) -> None:
        self.user_delta.pop(user_id, None)
