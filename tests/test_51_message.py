# tests/test_51_message.py

from core.classifier import Classifier
from core.embedder import Embedder
from core.decision import decide
from core.dwell import LabelDwell


def test_real_long_conversation():
    embedder = Embedder()
    clf = Classifier("models/lr.joblib")
    dwell = LabelDwell()

    user = "real_chat_user"

    # A realistic 100-turn conversation with continuity
    CHAT = [
        # ---- warmup / social noise ----
        ("hey", "SILENT"),
        ("hello", "SILENT"),
        ("hmm", "SILENT"),
        ("how are you", "SILENT"),
        ("yeah", "SILENT"),

        # ---- intent entry ----
        ("lets work on backend", "REQUEST"),
        ("auth system", "REQUEST"),
        ("login and signup", "REQUEST"),

        # ---- confirmation / acknowledgment loop ----
        ("cool", "REQUEST"),
        ("awesome", "REQUEST"),
        ("nice", "REQUEST"),
        ("ok", "REQUEST"),
        ("sounds good", "REQUEST"),
        ("yup", "REQUEST"),
        ("right", "REQUEST"),
        ("fine", "REQUEST"),

        # ---- concrete continuation ----
        ("should we use jwt", "REQUEST"),
        ("or session cookies", "REQUEST"),
        ("what do you recommend", "REQUEST"),

        # ---- mild hesitation ----
        ("hmm not sure", "REQUEST"),
        ("this feels complex", "REQUEST"),
        ("maybe start simple", "REQUEST"),

        # ---- steering but not reset ----
        ("makes sense", "REQUEST"),
        ("lets do that", "REQUEST"),

        # ---- implementation depth ----
        ("sveltekit auth flow", "REQUEST"),
        ("java backend api", "REQUEST"),
        ("token validation", "REQUEST"),
        ("refresh token", "REQUEST"),

        # ---- small noise inside task ----
        ("ok", "REQUEST"),
        ("cool", "REQUEST"),
        ("nice", "REQUEST"),

        # ---- clarification ----
        ("can you explain middleware", "REQUEST"),
        ("how does session storage work", "REQUEST"),

        # ---- frustration but not hostile ----
        ("this is confusing", "REQUEST"),
        ("im missing something", "REQUEST"),

        # ---- recovery ----
        ("lets slow down", "REQUEST"),
        ("step by step", "REQUEST"),

        # ---- topic reset explicitly ----
        ("change topic", "TOPIC_RESET"),
        ("lets talk about something else", "TOPIC_RESET"),

        # ---- post-reset noise ----
        ("hmm", "SILENT"),
        ("ok", "SILENT"),
        ("yeah", "SILENT"),

        # ---- new intent ----
        ("actually interview prep", "REQUEST"),
        ("system design questions", "REQUEST"),

        # ---- ack loop again ----
        ("cool", "REQUEST"),
        ("nice", "REQUEST"),
        ("sounds good", "REQUEST"),

        # ---- continuation ----
        ("scalability basics", "REQUEST"),
        ("load balancing", "REQUEST"),

        # ---- hostility boundary ----
        ("this is annoying", "REQUEST"),
        ("you are dumb", "HOSTILE"),
        ("go to hell", "HOSTILE"),
    ]

    for i, (text, expected) in enumerate(CHAT):
        vec = embedder.encode(text)
        label, conf, act = clf.predict(vec, user)
        label = dwell.apply(user, label, act)
        label = decide(label, conf)

        print(f"{i:02d} | {text!r} → {label} (conf={conf:.2f}, act={act:.2f})")
        assert label == expected
