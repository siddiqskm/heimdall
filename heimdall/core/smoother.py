# heimdall/core/smoother.py


from heimdall.core.types import Label


class ConfidenceSmoother:
    def __init__(self, alpha: float = 0.7) -> None:
        self.alpha = alpha
        self.state: dict[tuple[str, Label], float] = {}

    def apply(self, user_id: str, label: Label, confidence: float) -> float:
        key = (user_id, label)

        if key not in self.state:
            self.state[key] = confidence
            return confidence

        prev = self.state[key]
        smoothed = self.alpha * confidence + (1 - self.alpha) * prev
        self.state[key] = smoothed

        return smoothed
