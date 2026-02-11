# heimdall/core/model_types.py

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class ProbabilisticClassifier(Protocol):
    classes_: Sequence[int] | Sequence[str]
    multi_class: str
    predict_proba: Callable[[Sequence[NDArray[np.float64]]], NDArray[np.float64]]

    def __getattr__(self, name: str) -> Any:
        """Fallback for pickled models that may miss some attributes"""
        raise AttributeError(f"Attribute {name} not found")

