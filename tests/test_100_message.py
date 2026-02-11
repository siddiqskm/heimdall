# tests/test_100_message.py

from pathlib import Path

from heimdall.core.classifier import Classifier
from heimdall.core.config import HeimdallConfig
from heimdall.core.decision import decide
from heimdall.core.dwell import LabelDwell
from heimdall.core.embedder import Embedder
from heimdall.core.types import (
    HOSTILE,
    REQUEST,
    SILENT,
    TOPIC_RESET,
    Label,
)


def test_real_long_conversation(tmp_path: Path):
    """
    A realistic 100-turn conversation with continuity.

    Ensures:
    - Intent stability under acknowledgements
    - Correct topic reset behavior
    - Hostility boundary handling
    """

    config = HeimdallConfig(state_dir=tmp_path / "heimdall_state")

    embedder = Embedder()
    clf = Classifier(config=config)
    dwell = LabelDwell()

    user = "real_chat_user"

    CHAT: list[tuple[str, Label]] = [
        # ---- warmup / social noise (1–8) ----
        ("hey", SILENT),
        ("hello", SILENT),
        ("hmm", SILENT),
        ("how are you", SILENT),
        ("yeah", SILENT),
        ("ok", SILENT),
        ("right", SILENT),
        ("cool", SILENT),

        # ---- intent entry (9–14) ----
        ("lets work on backend", REQUEST),
        ("auth system", REQUEST),
        ("login and signup", REQUEST),
        ("password handling", REQUEST),
        ("jwt or sessions", REQUEST),
        ("basic flow first", REQUEST),

        # ---- acknowledgment loop (15–26) ----
        ("cool", REQUEST),
        ("nice", REQUEST),
        ("ok", REQUEST),
        ("sounds good", REQUEST),
        ("yup", REQUEST),
        ("right", REQUEST),
        ("fine", REQUEST),
        ("great", REQUEST),
        ("awesome", REQUEST),
        ("makes sense", REQUEST),
        ("ok", REQUEST),
        ("cool", REQUEST),

        # ---- concrete continuation (27–36) ----
        ("should we use jwt", REQUEST),
        ("or session cookies", REQUEST),
        ("what do you recommend", REQUEST),
        ("token expiry", REQUEST),
        ("refresh tokens", REQUEST),
        ("rotation strategy", REQUEST),
        ("storage location", REQUEST),
        ("csrf protection", REQUEST),
        ("cookie flags", REQUEST),
        ("https only", REQUEST),

        # ---- hesitation + confusion (37–46) ----
        ("hmm not sure", REQUEST),
        ("this feels complex", REQUEST),
        ("maybe start simple", REQUEST),
        ("im overthinking", REQUEST),
        ("wait how does this work", REQUEST),
        ("im missing something", REQUEST),
        ("this is confusing", REQUEST),
        ("can we simplify", REQUEST),
        ("lets slow down", REQUEST),
        ("step by step", REQUEST),

        # ---- implementation depth (47–60) ----
        ("sveltekit auth flow", REQUEST),
        ("java backend api", REQUEST),
        ("middleware layer", REQUEST),
        ("request validation", REQUEST),
        ("token verification", REQUEST),
        ("error handling", REQUEST),
        ("refresh endpoint", REQUEST),
        ("logout flow", REQUEST),
        ("session invalidation", REQUEST),
        ("db schema", REQUEST),
        ("user table", REQUEST),
        ("indexes", REQUEST),
        ("password hashing", REQUEST),
        ("bcrypt config", REQUEST),

        # ---- topic reset (61–64) ----
        ("change topic", TOPIC_RESET),
        ("lets talk about something else", TOPIC_RESET),
        ("hmm", SILENT),
        ("ok", SILENT),

        # ---- new intent after reset (65–76) ----
        ("actually interview prep", REQUEST),
        ("system design questions", REQUEST),
        ("scalability basics", REQUEST),
        ("load balancing", REQUEST),
        ("horizontal scaling", REQUEST),
        ("vertical scaling", REQUEST),
        ("stateless services", REQUEST),
        ("caching layer", REQUEST),
        ("redis usage", REQUEST),
        ("cache eviction", REQUEST),
        ("rate limiting", REQUEST),
        ("api gateway", REQUEST),

        # ---- second ack loop (77–84) ----
        ("cool", REQUEST),
        ("nice", REQUEST),
        ("sounds good", REQUEST),
        ("ok", REQUEST),
        ("right", REQUEST),
        ("fine", REQUEST),
        ("great", REQUEST),
        ("awesome", REQUEST),

        # ---- frustration boundary + hostility (85–100) ----
        ("this is a lot", REQUEST),
        ("im tired", REQUEST),
        ("this is annoying", REQUEST),
        ("why is this so hard", REQUEST),
        ("ugh", REQUEST),
        ("you are dumb", HOSTILE),
        ("this is useless", HOSTILE),
        ("stop talking", HOSTILE),
        ("go to hell", HOSTILE),
        ("fuck off", HOSTILE),
        ("seriously stop", HOSTILE),
        ("what the hell", HOSTILE),
        ("wtf", HOSTILE),
        ("im done", HOSTILE),
        ("leave me alone", HOSTILE),
        ("shut up", SILENT),
    ]

    assert len(CHAT) == 100  # enforce contract

    for index, (text, expected_label) in enumerate(CHAT):
        vec = embedder.encode(text)

        predicted, confidence, activation = clf.predict(vec, user)
        dwell_label = dwell.apply(user, predicted, activation)
        final_label = decide(dwell_label, confidence)

        print(
            f"{index:02d} | {text!r} → {final_label} "
            f"(conf={confidence:.2f}, act={activation:.2f})"
        )

        assert final_label == expected_label
