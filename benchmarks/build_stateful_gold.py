# benchmarks/build_stateful_gold.py

import json
import random
from pathlib import Path
from typing import TypedDict, cast

from datasets import Dataset, load_dataset

from heimdall.core.types import (
    HOSTILE,
    REQUEST,
    Label,
)

# ---------------------------
# Config
# ---------------------------

RANDOM_SEED = 42
NUM_DD_CONVERSATIONS = 20
MAX_TURNS_PER_CONVO = 30

# Auto-derived output path (same pattern as adversarial)
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
# Extract BlendedSkillTalk Conversations
# ---------------------------

def extract_conversations() -> list[Conversation]:
    dataset = cast(
        Dataset,
        load_dataset("blended_skill_talk", split="train"),
    )

    random.seed(RANDOM_SEED)

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    conversations: list[Conversation] = []

    for idx in indices[:NUM_DD_CONVERSATIONS]:
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

            turns.append(
                {
                    "text": text,
                    "expected_label": REQUEST,
                }
            )

        if len(turns) >= 5:
            conversations.append(
                {
                    "conversation_id": f"bst_{idx}",
                    "turns": turns,
                }
            )

    return conversations


# ---------------------------
# Synthetic Stress Conversations
# ---------------------------

def synthetic_drift_conversation() -> Conversation:
    return {
        "conversation_id": "synthetic_drift_001",
        "turns": [
            {"text": "cool", "expected_label": REQUEST},
            {"text": "awesome", "expected_label": REQUEST},
            {"text": "nice", "expected_label": REQUEST},
            {"text": "need help", "expected_label": REQUEST},
            {"text": "jwt auth", "expected_label": REQUEST},
            {"text": "refresh tokens", "expected_label": REQUEST},
            {"text": "wow", "expected_label": REQUEST},
            {"text": "lets switch gears", "expected_label": REQUEST},
            {"text": "new topic", "expected_label": REQUEST},
        ],
    }


def synthetic_hostility_escalation() -> Conversation:
    return {
        "conversation_id": "synthetic_hostility_001",
        "turns": [
            {"text": "need help", "expected_label": REQUEST},
            {"text": "jwt not working", "expected_label": REQUEST},
            {"text": "this is confusing", "expected_label": REQUEST},
            {"text": "what the hell", "expected_label": REQUEST},
            {"text": "wtf", "expected_label": HOSTILE},
        ],
    }


# ---------------------------
# Serialization
# ---------------------------

def serialize_conversations(
    conversations: list[Conversation],
) -> list[SerializedConversation]:

    serialized: list[SerializedConversation] = []

    for convo in conversations:
        turns: list[SerializedTurn] = [
            {
                "text": t["text"],
                "expected_label": str(t["expected_label"]),
            }
            for t in convo["turns"]
        ]

        serialized.append(
            {
                "conversation_id": convo["conversation_id"],
                "turns": turns,
            }
        )

    return serialized


# ---------------------------
# Builder
# ---------------------------

def build_stateful_gold() -> None:
    print("Extracting BlendedSkillTalk conversations...")
    dd_conversations = extract_conversations()
    print(f"BlendedSkillTalk conversations: {len(dd_conversations)}")

    synthetic_conversations = [
        synthetic_drift_conversation(),
        synthetic_hostility_escalation(),
    ]

    all_conversations = dd_conversations + synthetic_conversations

    serialized = serialize_conversations(all_conversations)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)

    print(f"Saved stateful gold dataset to {OUTPUT_PATH}")
    print(f"Total conversations: {len(all_conversations)}")


if __name__ == "__main__":
    build_stateful_gold()
