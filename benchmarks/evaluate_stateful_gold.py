# benchmarks/evaluate_stateful_gold.py

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
USER_ID = "__stateful_eval__"


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

class SerializedTurn(TypedDict):
    text: str
    expected_label: str


class SerializedConversation(TypedDict):
    conversation_id: str
    turns: list[SerializedTurn]


class Turn(TypedDict):
    text: str
    expected_label: Label


class Conversation(TypedDict):
    conversation_id: str
    turns: list[Turn]


# ---------------------------
# Loader
# ---------------------------

def load_gold() -> list[Conversation]:
    gold_path = build_gold_path()

    if not gold_path.exists():
        raise FileNotFoundError(
            f"Gold dataset not found at {gold_path}. "
            "Run the corresponding builder first."
        )

    with gold_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    conversations: list[Conversation] = []

    for convo in cast(list[SerializedConversation], raw):
        turns: list[Turn] = [
            {
                "text": t["text"],
                "expected_label": cast(Label, t["expected_label"]),
            }
            for t in convo["turns"]
        ]

        conversations.append(
            {
                "conversation_id": convo["conversation_id"],
                "turns": turns,
            }
        )

    return conversations


# ---------------------------
# Evaluator
# ---------------------------

def main() -> None:
    print("Loading stateful gold...")
    conversations = load_gold()
    print(f"Conversations: {len(conversations)}")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    clf = Classifier()

    total = 0
    correct = 0

    per_class_total: defaultdict[Label, int] = defaultdict(int)
    per_class_correct: defaultdict[Label, int] = defaultdict(int)

    confusion: defaultdict[Label, defaultdict[Label, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for convo in conversations:
        print(f"\n--- Conversation: {convo['conversation_id']} ---")

        clf.reset_user(USER_ID)
        clf.end_session()

        for turn in convo["turns"]:
            text = turn["text"]
            expected = turn["expected_label"]

            vector = embedder.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )

            pred_label, confidence, _ = clf.predict(
                vector,
                USER_ID,
            )

            clf.maybe_add_prototype(pred_label, vector, confidence)

            total += 1
            per_class_total[expected] += 1
            confusion[expected][pred_label] += 1

            if pred_label == expected:
                correct += 1
                per_class_correct[expected] += 1

            print(
                f"{text[:30]:30s} "
                f"→ Pred: {pred_label:12s} "
                f"Conf: {confidence:.3f}"
            )

    # ---------------------------
    # Metrics
    # ---------------------------

    accuracy = correct / total if total else 0.0

    print("\n=== STATEFUL RESULTS ===")
    print(f"Total turns: {total}")
    print(f"Accuracy: {accuracy:.4f}")

    print("\n=== PER-CLASS RECALL ===")
    for label in sorted(per_class_total.keys()):
        total_label = per_class_total[label]
        correct_label = per_class_correct[label]
        recall = correct_label / total_label if total_label else 0.0

        print(
            f"{label:12s} "
            f"count={total_label:4d} "
            f"recall={recall:.4f}"
        )

    print("\n=== CONFUSION MATRIX ===")
    labels = sorted(per_class_total.keys())

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
