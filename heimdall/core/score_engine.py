# heimdall/core/score_engine.py

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from heimdall.core.config import HeimdallConfig
from heimdall.core.prototypes import PrototypeStore, cosine_similarity
from heimdall.core.types import HOSTILE, TOPIC_RESET, Label

EmbeddingVector = NDArray[np.float64]


@dataclass(frozen=True)
class Scores:
    hostile_score: float
    reset_score: float
    utility_score: float


class ScoreEngine:

    def __init__(self, config: HeimdallConfig) -> None:
        self.config = config

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def compute(
        self,
        *,
        vector: EmbeddingVector,
        text: str,
        recent_vectors: list[EmbeddingVector],
        offline_prototypes: PrototypeStore,
        user_prototypes: PrototypeStore,
        session_prototypes: PrototypeStore,
    ) -> Scores:

        hostile_score = self._hostile_score(
            vector,
            offline_prototypes,
            user_prototypes,
            session_prototypes,
        )

        reset_score = self._reset_score(
            vector,
            recent_vectors,
            offline_prototypes,
            user_prototypes,
            session_prototypes,
        )

        utility_score = self._utility_score(
            vector,
            text,
            recent_vectors,
        )

        return Scores(
            hostile_score=hostile_score,
            reset_score=reset_score,
            utility_score=utility_score,
        )

    # ----------------------------------------------------------
    # Hostile
    # ----------------------------------------------------------

    def _hostile_score(
        self,
        vector: EmbeddingVector,
        offline: PrototypeStore,
        user: PrototypeStore,
        session: PrototypeStore,
    ) -> float:

        return max(
            self._max_similarity_for_label(vector, offline, HOSTILE),
            self._max_similarity_for_label(vector, user, HOSTILE),
            self._max_similarity_for_label(vector, session, HOSTILE),
        )

    # ----------------------------------------------------------
    # Reset
    # ----------------------------------------------------------

    def _reset_score(
        self,
        vector: EmbeddingVector,
        recent_vectors: list[EmbeddingVector],
        offline: PrototypeStore,
        user: PrototypeStore,
        session: PrototypeStore,
    ) -> float:

        phrase_score = max(
            self._max_similarity_for_label(vector, offline, TOPIC_RESET),
            self._max_similarity_for_label(vector, user, TOPIC_RESET),
            self._max_similarity_for_label(vector, session, TOPIC_RESET),
        )

        drift_score = self._drift_score(vector, recent_vectors)

        return max(
            phrase_score,
            drift_score * self.config.drift_weight,
        )

    def _drift_score(
        self,
        vector: EmbeddingVector,
        recent_vectors: list[EmbeddingVector],
    ) -> float:

        if not recent_vectors:
            return 0.0

        max_sim = max(
            cosine_similarity(vector, prev)
            for prev in recent_vectors
        )

        return 1.0 - max_sim

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------

    def _utility_score(
        self,
        vector: EmbeddingVector,
        text: str,
        recent_vectors: list[EmbeddingVector],
    ) -> float:

        novelty = self._novelty_score(vector, recent_vectors)
        info_density = self._info_density(text)
        richness = self._lexical_richness(text)

        return (
            self.config.novelty_weight * novelty
            + self.config.info_density_weight * info_density
            + self.config.richness_weight * richness
        )

    def _novelty_score(
        self,
        vector: EmbeddingVector,
        recent_vectors: list[EmbeddingVector],
    ) -> float:

        if not recent_vectors:
            return 1.0

        max_sim = max(
            cosine_similarity(vector, prev)
            for prev in recent_vectors
        )

        return 1.0 - max_sim

    def _info_density(self, text: str) -> float:
        tokens = text.strip().split()
        length = len(tokens)
        if length == 0:
            return 0.0
        return min(length / 20.0, 1.0)

    def _lexical_richness(self, text: str) -> float:
        tokens = text.strip().split()
        if not tokens:
            return 0.0
        unique = len(set(tokens))
        return unique / len(tokens)

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _max_similarity_for_label(
        self,
        vector: EmbeddingVector,
        store: PrototypeStore,
        label: Label,
    ) -> float:

        vectors = store.store.get(label)
        if not vectors:
            return 0.0

        return max(
            cosine_similarity(vector, proto)
            for proto in vectors
        )
