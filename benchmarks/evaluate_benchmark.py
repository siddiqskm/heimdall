# benchmarks/evaluate_benchmark.py

import json
from collections import defaultdict
from pathlib import Path
from typing import TypedDict, cast

from sentence_transformers import SentenceTransformer

from heimdall.core.classifier import Classifier
from heimdall.core.types import Label

# ---------------------------
# Config
# ---------------------------

VERSION = 1
DATA_FILE = "benchmark.json"
USER_ID = "__benchmark_eval__"


# ---------------------------
# Path Builder
# ---------------------------

def build_path() -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir / "data" / DATA_FILE


# ---------------------------
# Schema
# ---------------------------

class BenchmarkNode(TypedDict):
    message_id: str
    text: str
    llm_label: str
    manual_override: str | None
    continuity_id: str
    counter: int
    source_theme: str
    source_file: str
    stream_id: str


class EvalNode(TypedDict):
    text: str
    expected_label: Label


# ---------------------------
# Loader
# ---------------------------

def load_benchmark() -> list[EvalNode]:
    path = build_path()

    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark dataset not found at {path}"
        )

    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    nodes: list[EvalNode] = []

    for item in cast(list[BenchmarkNode], raw):
        # Manual override wins if present
        gold_label = (
            item["manual_override"]
            if item["manual_override"] is not None
            else item["llm_label"]
        )

        nodes.append(
            {
                "text": item["text"],
                "expected_label": cast(Label, gold_label),
            }
        )

    return nodes


# ---------------------------
# Evaluator
# ---------------------------

def main() -> None:
    print("Loading benchmark dataset...")
    nodes = load_benchmark()
    print(f"Total samples: {len(nodes)}")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    clf = Classifier()

    total = 0
    correct = 0

    per_class_total: defaultdict[Label, int] = defaultdict(int)
    per_class_correct: defaultdict[Label, int] = defaultdict(int)

    confusion: defaultdict[Label, defaultdict[Label, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for node in nodes:
        text = node["text"]
        expected = node["expected_label"]

        vector = embedder.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        pred_label, confidence, _ = clf.predict(
            vector,
            USER_ID,
        )

        total += 1
        per_class_total[expected] += 1
        confusion[expected][pred_label] += 1

        if pred_label == expected:
            correct += 1
            per_class_correct[expected] += 1

        print(
            f"{text[:40]:40s} "
            f"→ Pred: {pred_label:12s} "
            f"Conf: {confidence:.3f}"
        )

    # ---------------------------
    # Metrics
    # ---------------------------

    accuracy = correct / total if total else 0.0

    print("\n=== BENCHMARK RESULTS ===")
    print(f"Total samples: {total}")
    print(f"Accuracy: {accuracy:.4f}")

    print("\n=== PER-CLASS RECALL ===")
    labels = sorted(per_class_total.keys())

    for label in labels:
        total_label = per_class_total[label]
        correct_label = per_class_correct[label]
        recall = correct_label / total_label if total_label else 0.0

        print(
            f"{label:12s} "
            f"count={total_label:4d} "
            f"recall={recall:.4f}"
        )

    print("\n=== CONFUSION MATRIX ===")

    header = " " * 12 + " ".join(f"{label:12s}" for label in labels)
    print(header)

    for true_label in labels:
        row = f"{true_label:12s}"
        for pred_label in labels:
            row += f"{confusion[true_label][pred_label]:12d}"
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
