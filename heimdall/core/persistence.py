# heimdall/core/persistence.py

import json
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

BiasVector = NDArray[np.float64]
PathLike = str | Path


def save_chat_delta(path: PathLike, bias: BiasVector) -> None:
    """Persist a single chat's bias vector."""
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(bias.tolist(), f)


def load_chat_delta(path: PathLike) -> BiasVector | None:
    """Load a chat's bias vector; returns None if missing."""
    path = Path(path)
    try:
        with open(path) as f:
            raw = json.load(f)
    except FileNotFoundError:
        return None
    return np.array(raw, dtype=np.float64)
