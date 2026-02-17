# heimdall/core/persistence.py

import json
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

BiasVector = NDArray[np.float64]
UserDelta = dict[str, BiasVector]
PathLike = str | Path


def save_user_delta(path: PathLike, user_delta: UserDelta) -> None:
    path = Path(path)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)

    serializable = {
        user_id: bias.tolist()
        for user_id, bias in user_delta.items()
    }
    with open(path, "w") as f:
        json.dump(serializable, f)


def load_user_delta(path: PathLike) -> UserDelta:
    path = Path(path)

    try:
        with open(path) as f:
            raw = json.load(f)
    except FileNotFoundError:
        return {}

    user_delta: UserDelta = {}
    for user_id, bias_list in raw.items():
        user_delta[user_id] = np.array(bias_list, dtype=np.float64)

    return user_delta
