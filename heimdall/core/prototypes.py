# core/prototypes.py


import numpy as np
from numpy.typing import NDArray

from heimdall.core.types import Label

# ---- types ----

EmbeddingVector = NDArray[np.float64]


# ---- helpers ----

def cosine_similarity(
    a: EmbeddingVector,
    b: EmbeddingVector,
) -> float:
    """
    Pure NumPy cosine similarity.
    Returns value in [-1.0, 1.0].
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---- prototype store ----

class PrototypeStore:
    def __init__(self, max_per_label: int) -> None:
        self.store: dict[Label, list[EmbeddingVector]] = {}
        self.max_per_label = max_per_label

    def match(
        self,
        vector: EmbeddingVector,
        threshold: float,
    ) -> tuple[Label | None, float]:
        best_label: Label | None = None
        best_sim: float = 0.0

        for label, vectors in self.store.items():
            for proto in vectors:
                sim = cosine_similarity(vector, proto)
                if sim > best_sim:
                    best_sim = sim
                    best_label = label

        if best_sim >= threshold:
            return best_label, best_sim  # activation

        return None, 0.0

    def add(self, label: Label, vector: EmbeddingVector) -> None:
        self.store.setdefault(label, []).append(vector)
        # keep only most recent N prototypes per label
        self.store[label] = self.store[label][-self.max_per_label:]

    def clear(self) -> None:
        self.store.clear()
