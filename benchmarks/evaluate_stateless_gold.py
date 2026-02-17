# benchmarks/evaluate_stateless_gold.py

import json
from collections import defaultdict
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from heimdall.core.classifier import Classifier
from heimdall.core.types import Label

# ---------------------------
# Config
# ---------------------------

VERSION = 1
USER_ID = "__stateless_eval__"


# ---------------------------
# Derived Gold Path
# ---------------------------

def build_gold_path() -> Path:
    script_name = Path(__file__).resolve()
    dataset_name = script_name.stem.replace("evaluate_", "")
    filename = f"heimdall_{dataset_name}_v{VERSION}.json"
    return script_name.parent / "data" / filename


# ---------------------------
# Gold Schema
# ---------------------------

class GoldSample(TypedDict):
    text: str
    expected_label: str
    source: str
    meta: dict[str, object]


def load_gold() -> list[GoldSample]:
    gold_path = build_gold_path()

    if not gold_path.exists():
        raise FileNotFoundError(
            f"Gold dataset not found at {gold_path}. "
            "Run the corresponding builder first."
        )

    with gold_path.open(encoding="utf-8") as f:
        return json.load(f)


# ---------------------------
# Evaluator
# ---------------------------

def main() -> None:
    print("Loading gold dataset...")
    gold = load_gold()
    print(f"Samples: {len(gold)}")

    print("Loading embedder...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading classifier...")
    clf = Classifier()

    total = 0
    correct = 0

    per_class_counts: defaultdict[Label, int] = defaultdict(int)
    per_class_correct: defaultdict[Label, int] = defaultdict(int)

    confusion: defaultdict[Label, defaultdict[Label, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    confidences: list[float] = []

    for sample in gold:
        text: str = sample["text"]
        expected = cast(Label, sample["expected_label"])

        # Stateless → reset per sample
        clf.reset_user(USER_ID)
        clf.end_session()

        vector: NDArray[np.float64] = embedder.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        pred_label, confidence, _ = clf.predict(vector, USER_ID)

        total += 1
        per_class_counts[expected] += 1
        confusion[expected][pred_label] += 1
        confidences.append(confidence)

        if pred_label == expected:
            correct += 1
            per_class_correct[expected] += 1

    accuracy = correct / total if total else 0.0

    print("\n=== STATELESS RESULTS ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Avg confidence: {np.mean(confidences):.4f}")

    print("\n=== PER CLASS ===")
    for label in sorted(per_class_counts.keys()):
        count = per_class_counts[label]
        correct_count = per_class_correct[label]
        recall = correct_count / count if count else 0.0

        print(
            f"{label:12s} "
            f"count={count:4d} "
            f"recall={recall:.4f}"
        )

    print("\n=== CONFUSION MATRIX ===")
    labels = sorted(per_class_counts.keys())

    header = " " * 12 + " ".join(f"{label:12s}" for label in labels)
    print(header)

    for true_label in labels:
        row = f"{true_label:12s}"
        for pred_label in labels:
            value = confusion[true_label][pred_label]
            row += f"{value:12d}"
        print(row)

    print("\n=== PER-CLASS METRICS ===")

    for label in labels:
        tp = confusion[label][label]

        fp = sum(
            confusion[other][label]
            for other in labels
            if other != label
        )

        fn = sum(
            confusion[label][other]
            for other in labels
            if other != label
        )

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        print(
            f"{label:12s} "
            f"precision={precision:.4f} "
            f"recall={recall:.4f} "
            f"f1={f1:.4f}"
        )


if __name__ == "__main__":
    main()
