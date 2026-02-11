# heimdall/core/model_loader.py

import json
from importlib.resources import files
from importlib.resources.abc import Traversable

import joblib
import numpy as np
from numpy.typing import NDArray

from heimdall.core.model_types import ProbabilisticClassifier
from heimdall.core.types import LABELS, Label

type EmbeddingVector = NDArray[np.float64]


def load_lr_model() -> ProbabilisticClassifier:
    model_path: Traversable = files("heimdall").joinpath("models/lr.joblib")
    return joblib.load(model_path)


def load_offline_prototypes() -> dict[Label, list[EmbeddingVector]]:
    """
    Load immutable offline prototypes bundled with the package.

    These are NOT runtime state and must not live in state_dir.
    """

    proto_path: Traversable = files("heimdall").joinpath(
        "state/prototypes_offline.json"
    )

    with proto_path.open("r") as f:
        raw: dict[str, list[list[float]]] = json.load(f)

    result: dict[Label, list[EmbeddingVector]] = {}

    for key, vectors in raw.items():
        # ---- TYPE NARROWING ----
        if key not in LABELS:
            raise ValueError(
                f"Invalid label '{key}' in prototypes_offline.json"
            )

        label: Label = key  # now safe — validated against LABELS

        result[label] = [
            np.array(vec, dtype=np.float64)
            for vec in vectors
        ]

    return result
