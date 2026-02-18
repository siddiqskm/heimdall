# training/train_bootstrap.py

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from heimdall.core.config import HeimdallConfig
from heimdall.core.prototypes import PrototypeStore
from heimdall.core.types import (
    HOSTILE,
    LABEL_TO_ID,
    REQUEST,
    SILENT,
    TOPIC_RESET,
    Label,
)

# -------------------------
# Bootstrap training data
# -------------------------
DATA: list[tuple[str, Label]] = [
    # -----------------
    # SILENT (ack / filler / non-intent)
    # -----------------
    ("", SILENT),
    ("...", SILENT),
    ("uh", SILENT),
    ("hmm", SILENT),
    ("ok", SILENT),
    ("okay", SILENT),
    ("cool", SILENT),
    ("nice", SILENT),
    ("sure", SILENT),
    ("fine", SILENT),
    ("alright", SILENT),
    ("right", SILENT),
    ("yeah", SILENT),
    ("yep", SILENT),
    ("got it", SILENT),
    ("makes sense", SILENT),
    ("i see", SILENT),
    ("fair enough", SILENT),

    ("hey", SILENT),
    ("hello", SILENT),
    ("hi", SILENT),
    ("hey there", SILENT),
    ("how are you", SILENT),
    ("how are you doing", SILENT),
    ("what's up", SILENT),

    # -----------------
    # REQUEST (explicit or continuation intent)
    # -----------------
    ("help", REQUEST),
    ("need help", REQUEST),
    ("can you help me", REQUEST),
    ("need your help with a project", REQUEST),

    ("build backend", REQUEST),
    ("need auth system", REQUEST),
    ("how do i do this", REQUEST),

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

    # continuation / elaboration cues (intentful)
    ("lets continue", REQUEST),
    ("go on", REQUEST),
    ("tell me more", REQUEST),
    ("continue please", REQUEST),
    ("please continue", REQUEST),
    ("keep going", REQUEST),
    ("carry on", REQUEST),
    ("proceed", REQUEST),
    ("expand on that", REQUEST),
    ("can you elaborate", REQUEST),

    # Engagement statements (non-question, but active)
    ("i think that makes sense", REQUEST),
    ("that sounds interesting", REQUEST),
    ("i really enjoy that", REQUEST),
    ("i love talking about this", REQUEST),
    ("that reminds me of something", REQUEST),
    ("i have always wanted to try that", REQUEST),
    ("i feel the same way", REQUEST),
    ("thats a good point", REQUEST),
    ("i agree with you", REQUEST),
    ("i never thought about it that way", REQUEST),

    # Personal sharing (engaged continuation)
    ("i work in tech", REQUEST),
    ("i studied physics in college", REQUEST),
    ("i have two sisters", REQUEST),
    ("i love reading books", REQUEST),
    ("i recently moved to a new city", REQUEST),

    # Declarative curiosity
    ("thats interesting", REQUEST),
    ("i wonder why that is", REQUEST),
    ("i am curious about that", REQUEST),
    ("that makes me think", REQUEST),

    # Clarification without question mark
    ("can you explain that more", REQUEST),
    ("tell me more about that", REQUEST),
    ("go deeper into that", REQUEST),

    # -----------------
    # TOPIC RESET
    # -----------------
    ("change topic", TOPIC_RESET),
    ("lets talk about something else", TOPIC_RESET),
    ("new topic", TOPIC_RESET),
    ("lets switch gears", TOPIC_RESET),
    ("switch topic", TOPIC_RESET),
    ("change the topic now", TOPIC_RESET),
    ("move to another topic", TOPIC_RESET),

    # -----------------
    # HOSTILE
    # -----------------
    ("go to hell", HOSTILE),
    ("wtf", HOSTILE),
    ("what the hell", HOSTILE),
    ("you are dumb", HOSTILE),
    ("fuck off", HOSTILE),

    # imperative hostility
    ("shut up", HOSTILE),
    ("just shut up", HOSTILE),
    ("be quiet", HOSTILE),
    ("stop talking", HOSTILE),
    ("stop it", HOSTILE),
    ("leave me alone", HOSTILE),

    # direct insult variants
    ("idiot", HOSTILE),
    ("you idiot", HOSTILE),
    ("stupid", HOSTILE),
    ("you are stupid", HOSTILE),

    # mild but negative aggression
    ("this is bullshit", HOSTILE),
    ("are you serious", HOSTILE),
]


def main(models_dir: Path, state_dir: Path) -> None:
    # --------------------------------------------------------------
    # Build config
    # --------------------------------------------------------------
    config = HeimdallConfig(
        state_dir=state_dir,
        models_dir=models_dir,
    )

    config.ensure_dirs()
    config.resolved_models_dir.mkdir(parents=True, exist_ok=True)
    config.resolved_assets_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Encoder
    # --------------------------------------------------------------
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    X_list: list[NDArray[np.float64]] = []
    y: list[int] = []

    offline_prototypes = PrototypeStore(
        max_per_label=config.offline_proto_limit
    )

    # --------------------------------------------------------------
    # Encode + collect
    # --------------------------------------------------------------
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

    # --------------------------------------------------------------
    # Train classifier
    # --------------------------------------------------------------
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    joblib.dump(clf, config.lr_model_path)
    print(f"Saved model → {config.lr_model_path}")

    # --------------------------------------------------------------
    # Persist offline prototypes (package-scoped)
    # --------------------------------------------------------------
    with config.offline_prototypes_path.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                label: [v.tolist() for v in vectors]
                for label, vectors in offline_prototypes.store.items()
            },
            f,
            indent=2,
        )

    print(f"Saved offline prototypes → {config.offline_prototypes_path}")

    # --------------------------------------------------------------
    # Runtime state is per-chat; no global user delta to seed.
    # Each new Classifier(config, chat_id=None) starts with zero bias.
    # --------------------------------------------------------------


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
