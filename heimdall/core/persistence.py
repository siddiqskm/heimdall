# core/persistence.py

import json
import os
from typing import Dict
import numpy as np
from numpy.typing import NDArray


BiasVector = NDArray[np.float64]
UserDelta = Dict[str, BiasVector]


def save_user_delta(path: str, user_delta: UserDelta) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    serializable = {
        user_id: bias.tolist()
        for user_id, bias in user_delta.items()
    }
    with open(path, "w") as f:
        json.dump(serializable, f)


def load_user_delta(path: str) -> UserDelta:
    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return {}

    user_delta: UserDelta = {}
    for user_id, bias_list in raw.items():
        user_delta[user_id] = np.array(bias_list, dtype=np.float64)

    return user_delta
