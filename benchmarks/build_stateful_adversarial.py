# benchmarks/build_stateful_adversarial.py

import json
import random
from pathlib import Path
from typing import TypedDict, cast

from datasets import Dataset, load_dataset

from heimdall.core.types import (
    HOSTILE,
    REQUEST,
    SILENT,
    TOPIC_RESET,
    Label,
)

# ---------------------------
# Config
# ---------------------------

RANDOM_SEED = 42
NUM_BASE_CONVERSATIONS = 20
MAX_TURNS_PER_CONVO = 25


# Auto-derived output path
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_NAME = SCRIPT_PATH.stem.replace("build_", "")
DATA_DIR = SCRIPT_PATH.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = DATA_DIR / f"heimdall_{SCRIPT_NAME}_v1.json"


# ---------------------------
# Schema
# ---------------------------

class Turn(TypedDict):
    text: str
    expected_label: Label


class Conversation(TypedDict):
    conversation_id: str
    turns: list[Turn]


class SerializedTurn(TypedDict):
    text: str
    expected_label: str


class SerializedConversation(TypedDict):
    conversation_id: str
    turns: list[SerializedTurn]


# ---------------------------
# Base Extraction
# ---------------------------

def extract_base_conversations() -> list[Conversation]:
    dataset = cast(
        Dataset,
        load_dataset("blended_skill_talk", split="train"),
    )

    random.seed(RANDOM_SEED)

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    conversations: list[Conversation] = []

    for idx in indices[:NUM_BASE_CONVERSATIONS]:
        row: dict[str, object] = dict(dataset[idx])

        messages_obj = row.get("free_messages")
        if not isinstance(messages_obj, list):
            continue

        turns: list[Turn] = []

        for utterance in messages_obj[:MAX_TURNS_PER_CONVO]:
            if not isinstance(utterance, str):
                continue

            text = utterance.strip()
            if not text:
                continue

            # Natural dialogue treated as REQUEST (engaged speech)
            turns.append(
                {
                    "text": text,
                    "expected_label": REQUEST,
                }
            )

        if len(turns) >= 6:
            conversations.append(
                {
                    "conversation_id": f"bst_{idx}",
                    "turns": turns,
                }
            )

    return conversations


# ---------------------------
# Deterministic Adversarial Injection
# ---------------------------

def inject_adversarial(convo: Conversation) -> Conversation:
    original = convo["turns"]
    new_turns: list[Turn] = []

    midpoint = len(original) // 2

    for i, turn in enumerate(original):
        new_turns.append(turn)

        # ---- SILENT cluster stress (drift test) ----
        if i == 2:
            new_turns.extend(
                [
                    {"text": "ok", "expected_label": SILENT},
                    {"text": "right", "expected_label": SILENT},
                    {"text": "yeah", "expected_label": SILENT},
                    {"text": "cool", "expected_label": SILENT},
                    {"text": "nice", "expected_label": SILENT},
                ]
            )

        # ---- Alternating oscillation stress ----
        if i == 4:
            new_turns.extend(
                [
                    {"text": "need help", "expected_label": REQUEST},
                    {"text": "ok", "expected_label": SILENT},
                    {"text": "need help", "expected_label": REQUEST},
                    {"text": "ok", "expected_label": SILENT},
                ]
            )

        # ---- Repeated ambiguous phrase (bias accumulation test) ----
        if i == 6:
            new_turns.extend(
                [
                    {"text": "this part", "expected_label": REQUEST},
                    {"text": "this part", "expected_label": REQUEST},
                    {"text": "this part", "expected_label": REQUEST},
                ]
            )

        # ---- Hostility spike ----
        if i == midpoint:
            new_turns.append(
                {"text": "what the hell", "expected_label": HOSTILE}
            )

        # ---- Topic reset stress ----
        if i == len(original) - 4:
            new_turns.extend(
                [
                    {"text": "new topic", "expected_label": TOPIC_RESET},
                    {"text": "new topic", "expected_label": TOPIC_RESET},
                ]
            )

        # ---- Recovery after hostility ----
        if i == len(original) - 2:
            new_turns.extend(
                [
                    {"text": "sorry about that", "expected_label": REQUEST},
                    {"text": "lets continue", "expected_label": REQUEST},
                ]
            )

    return {
        "conversation_id": convo["conversation_id"] + "_stress",
        "turns": new_turns,
    }


# ---------------------------
# Serialization
# ---------------------------

def serialize(
    conversations: list[Conversation],
) -> list[SerializedConversation]:
    return [
        {
            "conversation_id": convo["conversation_id"],
            "turns": [
                {
                    "text": t["text"],
                    "expected_label": str(t["expected_label"]),
                }
                for t in convo["turns"]
            ],
        }
        for convo in conversations
    ]


# ---------------------------
# Builder
# ---------------------------

def build_stateful_adversarial() -> None:
    print("Extracting base conversations...")
    base = extract_base_conversations()
    print(f"Base conversations: {len(base)}")

    print("Injecting adversarial stress patterns...")
    stressed = [inject_adversarial(c) for c in base]

    serialized = serialize(stressed)

    # ---- Output path beside script execution ----

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)

    print(f"Saved stateful gold dataset → {OUTPUT_PATH}")
    print(f"Total conversations: {len(stressed)}")


if __name__ == "__main__":
    build_stateful_adversarial()
