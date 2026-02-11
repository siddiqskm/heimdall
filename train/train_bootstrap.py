# train/train_bootstrap.py

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from heimdall.core.prototypes import PrototypeStore
from heimdall.core.types import (
    HOSTILE,
    LABEL_TO_ID,
    REQUEST,
    SILENT,
    STEER,
    TOPIC_RESET,
    Label,
)

# -------------------------
# Bootstrap training data
# -------------------------
DATA: list[tuple[str, Label]] = [
    ("", SILENT),
    ("...", SILENT),
    ("uh", SILENT),
    ("hmm", SILENT),
    ("ok", SILENT),
    ("cool", SILENT),
    ("nice", SILENT),

    ("hey", SILENT),
    ("hello", SILENT),
    ("hi", SILENT),
    ("hey there", SILENT),
    ("how are you", SILENT),
    ("how are you doing", SILENT),
    ("what's up", SILENT),

    ("okay", STEER),
    ("sounds good", STEER),
    ("awesome", STEER),
    ("right", STEER),
    ("fine", STEER),

    ("help", REQUEST),
    ("need help", REQUEST),
    ("can you help me", REQUEST),
    ("need your help with a project", REQUEST),

    ("build backend", REQUEST),
    ("need auth system", REQUEST),
    ("how do I do this", REQUEST),

    ("lets talk about textiles", REQUEST),
    ("can we discuss art", REQUEST),
    ("suggest a topic", REQUEST),
    ("what should we discuss", REQUEST),
    ("any ideas", REQUEST),
    ("tell me something interesting", REQUEST),
    ("surprise me", REQUEST),

    ("any suggestions to discuss?", REQUEST),
    ("any suggestions?", REQUEST),
    ("what can we discuss?", REQUEST),
    ("what do you suggest we talk about?", REQUEST),
    ("give me something to discuss", REQUEST),
    ("pick a topic for discussion", REQUEST),

    ("change topic", TOPIC_RESET),
    ("lets talk about something else", TOPIC_RESET),
    ("new topic", TOPIC_RESET),
    ("lets switch gears", TOPIC_RESET),
    ("switch topic", TOPIC_RESET),
    ("change the topic now", TOPIC_RESET),

    ("go to hell", HOSTILE),
    ("wtf", HOSTILE),
    ("what the hell", HOSTILE),
    ("you are dumb", HOSTILE),
    ("fuck off", HOSTILE),
]


def main(models_dir: Path, state_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Model + storage setup
    # -------------------------
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    X_list: list[NDArray[np.float64]] = []
    y: list[int] = []

    offline_prototypes = PrototypeStore(max_per_label=50)

    # -------------------------
    # Encode + collect
    # -------------------------
    for text, label in DATA:
        embedding: NDArray[np.float64] = encoder.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        X_list.append(embedding)
        y.append(LABEL_TO_ID[label])

        offline_prototypes.add(label, embedding)

    X: NDArray[np.float64] = np.vstack(X_list)

    # -------------------------
    # Train classifier
    # -------------------------
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    model_path = models_dir / "lr.joblib"
    joblib.dump(clf, model_path)
    print(f"Saved {model_path}")

    # -------------------------
    # Persist offline prototypes
    # -------------------------
    proto_path = state_dir / "prototypes_offline.json"
    with proto_path.open("w") as f:
        json.dump(
            {
                label: [v.tolist() for v in vectors]
                for label, vectors in offline_prototypes.store.items()
            },
            f,
            indent=2,
        )
    print(f"Saved {proto_path}")

    # -------------------------
    # Seed LearningGate state
    # -------------------------
    user_delta_path = state_dir / "user_delta.json"
    initial_user_delta = {
        "__global__": [0.0 for _ in range(len(LABEL_TO_ID))]
    }
    with user_delta_path.open("w") as f:
        json.dump(initial_user_delta, f, indent=2)
    print(f"Initialized {user_delta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Heimdall bootstrap training")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("heimdall/models"),
        help="Directory to write trained models",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("state"),
        help="Directory to write state JSON files",
    )

    args = parser.parse_args()
    main(args.models_dir, args.state_dir)
