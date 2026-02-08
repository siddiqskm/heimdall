# train_bootstrap.py

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np
from numpy.typing import NDArray
import json
import os

from core.types import (
    Label,
    LABEL_TO_ID,
    SILENT,
    STEER,
    REQUEST,
    TOPIC_RESET,
    HOSTILE,
)
from core.prototypes import PrototypeStore


# -------------------------
# Bootstrap training data
# -------------------------
DATA: list[tuple[str, Label]] = [
    # -----------------
    # SILENT (non-actionable noise)
    # -----------------
    ("", SILENT),
    ("...", SILENT),
    ("uh", SILENT),
    ("hmm", SILENT),
    ("ok", SILENT),
    ("cool", SILENT),
    ("nice", SILENT),

    # -----------------
    # SILENT (social greetings / noise)
    # -----------------
    ("hey", SILENT),
    ("hello", SILENT),
    ("hi", SILENT),
    ("hey there", SILENT),
    ("how are you", SILENT),
    ("how are you doing", SILENT),
    ("what's up", SILENT),

    # -----------------
    # STEER (ack / continuation without intent)
    # -----------------
    ("okay", STEER),
    ("sounds good", STEER),
    ("awesome", STEER),
    ("right", STEER),
    ("fine", STEER),

    # -----------------
    # REQUEST (explicit + exploratory intent)
    # -----------------
    ("help", REQUEST),
    ("need help", REQUEST),
    ("can you help me", REQUEST),
    ("need your help with a project", REQUEST),

    ("build backend", REQUEST),
    ("need auth system", REQUEST),
    ("how do I do this", REQUEST),

    # exploratory / topic-driven (IMPORTANT)
    ("lets talk about textiles", REQUEST),
    ("can we discuss art", REQUEST),
    ("suggest a topic", REQUEST),
    ("what should we discuss", REQUEST),
    ("any ideas", REQUEST),
    ("tell me something interesting", REQUEST),
    ("surprise me", REQUEST),

    # exploratory / topic-driven (IMPORTANT)
    ("any suggestions to discuss?", REQUEST),
    ("any suggestions?", REQUEST),
    ("what can we discuss?", REQUEST),
    ("what do you suggest we talk about?", REQUEST),
    ("give me something to discuss", REQUEST),
    ("pick a topic for discussion", REQUEST),

    # -----------------
    # TOPIC_RESET
    # -----------------
    ("change topic", TOPIC_RESET),
    ("lets talk about something else", TOPIC_RESET),
    ("new topic", TOPIC_RESET),
    ("lets switch gears", TOPIC_RESET),
    ("switch topic", TOPIC_RESET),
    ("change the topic now", TOPIC_RESET),

    # -----------------
    # HOSTILE (anger / profanity / directed frustration)
    # -----------------
    ("go to hell", HOSTILE),
    ("wtf", HOSTILE),
    ("what the hell", HOSTILE),
    ("what the hell?", HOSTILE),
    ("you are dumb", HOSTILE),
    ("fuck off", HOSTILE),
]


# -------------------------
# Model + storage setup
# -------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

X_list: list[NDArray[np.float64]] = []
y: list[int] = []

offline_prototypes = PrototypeStore(max_per_label=50)

# -------------------------
# Encode + collect
# -------------------------
for text, label in DATA:
    embedding: NDArray[np.float64] = model.encode(
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

os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/lr.joblib")
print("Saved models/lr.joblib")

# -------------------------
# Persist offline prototypes
# -------------------------
os.makedirs("state", exist_ok=True)

with open("state/prototypes_offline.json", "w") as f:
    json.dump(
        {label: [v.tolist() for v in vectors]
         for label, vectors in offline_prototypes.store.items()},
        f,
        indent=2,
    )

print("Saved state/prototypes_offline.json")

# -------------------------
# Seed LearningGate state
# -------------------------
# user_delta.json should NEVER be empty
initial_user_delta = {
    "__global__": [0.0 for _ in range(len(LABEL_TO_ID))]
}

with open("state/user_delta.json", "w") as f:
    json.dump(initial_user_delta, f, indent=2)

print("Initialized state/user_delta.json")
