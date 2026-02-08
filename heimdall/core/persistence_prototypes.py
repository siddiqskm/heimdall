# core/persistence_prototypes.py

import json
import numpy as np
from typing import Dict, List
from numpy.typing import NDArray

from core.types import Label

EmbeddingVector = NDArray[np.float64]


def load_prototypes(path: str) -> Dict[Label, List[EmbeddingVector]]:
    try:
        with open(path, "r") as f:
            raw = json.load(f)
        return {
            label: [np.array(v, dtype=np.float64) for v in vectors]
            for label, vectors in raw.items()
        }
    except FileNotFoundError:
        return {}


def save_prototypes(
    path: str,
    store: Dict[Label, List[EmbeddingVector]],
) -> None:
    serializable = {
        label: [v.tolist() for v in vectors]
        for label, vectors in store.items()
    }
    with open(path, "w") as f:
        json.dump(serializable, f)
