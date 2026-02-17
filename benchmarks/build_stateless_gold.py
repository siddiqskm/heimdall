# benchmarks/build_stateless_gold.py

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, load_dataset

from heimdall.core.types import (
    REQUEST,
    SILENT,
    Label,
)

# ---------------------------
# Config
# ---------------------------

RANDOM_SEED = 42
SAMPLES_PER_LABEL = 150
GOEMOTIONS_PER_LABEL = 75
VERSION = 1


# ---------------------------
# Derived Paths
# ---------------------------
# Auto-derived output path
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_NAME = SCRIPT_PATH.stem.replace("build_", "")
DATA_DIR = SCRIPT_PATH.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = DATA_DIR / f"heimdall_{SCRIPT_NAME}_v1.json"


# ---------------------------
# Minimal heuristic mapping
# ---------------------------

SILENT_PHRASES = {
    "ok", "okay", "yeah", "yep", "cool",
    "nice", "sure", "thanks", "alright",
    "hmm", "uh", "idk"
}


def map_text_to_label(text: str) -> Label:
    lowered = text.lower().strip()

    if lowered in SILENT_PHRASES:
        return SILENT

    return REQUEST


# ---------------------------
# BlendedSkillTalk
# ---------------------------

def extract_bst_samples() -> list[dict[str, Any]]:
    dataset = cast(
        Dataset,
        load_dataset("blended_skill_talk", split="train"),
    )

    buckets: defaultdict[Label, list[dict[str, Any]]] = defaultdict(list)

    for conv_id, item in enumerate(dataset):
        row: dict[str, Any] = dict(item)
        messages = row.get("free_messages")

        if not isinstance(messages, list):
            continue

        for turn_id, utterance in enumerate(messages):
            if not isinstance(utterance, str):
                continue

            text = utterance.strip()
            if not text:
                continue

            label = map_text_to_label(text)

            buckets[label].append(
                {
                    "text": text,
                    "expected_label": label,
                    "source": "blended_skill_talk",
                    "meta": {
                        "conversation_id": conv_id,
                        "turn_id": turn_id,
                    },
                }
            )

    final: list[dict[str, Any]] = []

    for label in (REQUEST, SILENT):
        items = buckets[label]
        if not items:
            continue

        chosen = random.sample(
            items,
            min(SAMPLES_PER_LABEL, len(items)),
        )
        final.extend(chosen)

    return final


# ---------------------------
# GoEmotions
# ---------------------------

def extract_goemotions_samples() -> list[dict[str, Any]]:
    dataset = cast(
        Dataset,
        load_dataset("go_emotions", split="train"),
    )

    labels_feature = dataset.features.get("labels")
    if labels_feature is None:
        raise ValueError("GoEmotions dataset missing 'labels' feature")

    label_names: list[str] = list(labels_feature.feature.names)

    target_emotions = {"anger", "annoyance", "confusion"}

    target_ids = [
        idx for idx, name in enumerate(label_names)
        if name in target_emotions
    ]

    collected: list[dict[str, Any]] = []

    for item in dataset:
        row: dict[str, Any] = dict(item)

        text_raw = row.get("text")
        labels = row.get("labels", [])

        if not isinstance(text_raw, str):
            continue

        text = text_raw.strip()
        if not text:
            continue

        if any(int(lid) in target_ids for lid in labels):
            collected.append(
                {
                    "text": text,
                    "expected_label": REQUEST,
                    "source": "go_emotions",
                    "meta": {},
                }
            )

    return random.sample(
        collected,
        min(GOEMOTIONS_PER_LABEL, len(collected)),
    )


# ---------------------------
# Builder
# ---------------------------

def build_stateless_gold() -> None:
    random.seed(RANDOM_SEED)

    print("Extracting BlendedSkillTalk samples...")
    bst_samples = extract_bst_samples()
    print(f"BST samples: {len(bst_samples)}")

    print("Extracting GoEmotions samples...")
    ge_samples = extract_goemotions_samples()
    print(f"GoEmotions samples: {len(ge_samples)}")

    all_samples = bst_samples + ge_samples

    print(f"\nTotal stateless gold samples: {len(all_samples)}")

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_stateless_gold()
