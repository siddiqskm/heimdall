# heimdall/core/model_loader.py

import json
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray

from heimdall.core.model_types import ProbabilisticClassifier
from heimdall.core.types import LABELS, Label

type EmbeddingVector = NDArray[np.float64]


def load_lr_model(model_path: Path) -> ProbabilisticClassifier:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}"
        )

    return joblib.load(model_path)


def load_offline_prototypes(
    proto_path: Path,
) -> dict[Label, list[EmbeddingVector]]:
    if not proto_path.exists():
        raise FileNotFoundError(
            f"Offline prototypes not found at {proto_path}"
        )

    with proto_path.open("r", encoding="utf-8") as f:
        raw: dict[str, list[list[float]]] = json.load(f)

    result: dict[Label, list[EmbeddingVector]] = {}

    for key, vectors in raw.items():
        if key not in LABELS:
            raise ValueError(
                f"Invalid label '{key}' in prototypes_offline.json"
            )

        label: Label = key

        result[label] = [
            np.array(vec, dtype=np.float64)
            for vec in vectors
        ]

    return result
