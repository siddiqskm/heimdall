# heimdall/core/model_types.py

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class ProbabilisticClassifier(Protocol):
    classes_: Sequence[int] | Sequence[str]
    multi_class: str

    def predict_proba(
        self,
        X: Sequence[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        ...

