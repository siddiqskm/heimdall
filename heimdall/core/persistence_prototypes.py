# heimdall/core/persistence_prototypes.py

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from heimdall.core.types import Label

EmbeddingVector = NDArray[np.float64]
type PathLike = str | Path


def load_prototypes(path: PathLike) -> dict[Label, list[EmbeddingVector]]:
    path = Path(path)

    try:
        with open(path) as f:
            raw = json.load(f)
        return {
            label: [np.array(v, dtype=np.float64) for v in vectors]
            for label, vectors in raw.items()
        }
    except FileNotFoundError:
        return {}


def save_prototypes(
    path: PathLike,
    store: dict[Label, list[EmbeddingVector]],
) -> None:
    serializable = {
        label: [v.tolist() for v in vectors]
        for label, vectors in store.items()
    }
    
    path = Path(path)

    with open(path, "w") as f:
        json.dump(serializable, f)
