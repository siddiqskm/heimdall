# tests/test_51_message.py

import logging
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

logger = logging.getLogger(__name__)


def test_real_long_conversation(tmp_path: Path):
    """
    Simulates a long, realistic conversation and ensures
    intent stability, topic reset behavior, and hostility boundaries.
    """

    config = HeimdallConfig(state_dir=tmp_path / ".heimdall")

    embedder = Embedder()
    clf = Classifier(config=config)
    dwell = LabelDwell(config=config, chat_id=clf.chat_id)

    CHAT: list[tuple[str, Label]] = [
        # ---- warmup / social noise ----
        ("hey", SILENT),
        ("hello", SILENT),
        ("hmm", SILENT),
        ("how are you", SILENT),
        ("yeah", SILENT),

        # ---- intent entry ----
        ("lets work on backend", REQUEST),
        ("auth system", REQUEST),
        ("login and signup", REQUEST),

        # ---- confirmation / acknowledgment loop ----
        ("cool", REQUEST),
        ("awesome", REQUEST),
        ("nice", REQUEST),
        ("ok", REQUEST),
        ("sounds good", REQUEST),
        ("yup", REQUEST),
        ("right", REQUEST),
        ("fine", REQUEST),

        # ---- concrete continuation ----
        ("should we use jwt", REQUEST),
        ("or session cookies", REQUEST),
        ("what do you recommend", REQUEST),

        # ---- mild hesitation ----
        ("hmm not sure", REQUEST),
        ("this feels complex", REQUEST),
        ("maybe start simple", REQUEST),

        # ---- steering but not reset ----
        ("makes sense", REQUEST),
        ("lets do that", REQUEST),

        # ---- implementation depth ----
        ("sveltekit auth flow", REQUEST),
        ("java backend api", REQUEST),
        ("token validation", REQUEST),
        ("refresh token", REQUEST),

        # ---- small noise inside task ----
        ("ok", REQUEST),
        ("cool", REQUEST),
        ("nice", REQUEST),

        # ---- clarification ----
        ("can you explain middleware", REQUEST),
        ("how does session storage work", REQUEST),

        # ---- frustration but not hostile ----
        ("this is confusing", REQUEST),
        ("im missing something", REQUEST),

        # ---- recovery ----
        ("lets slow down", REQUEST),
        ("step by step", REQUEST),

        # ---- topic reset explicitly ----
        ("change topic", TOPIC_RESET),
        ("lets talk about something else", TOPIC_RESET),

        # ---- post-reset noise ----
        ("hmm", SILENT),
        ("ok", SILENT),
        ("yeah", SILENT),

        # ---- new intent ----
        ("actually interview prep", REQUEST),
        ("system design questions", REQUEST),

        # ---- ack loop again ----
        ("cool", REQUEST),
        ("nice", REQUEST),
        ("sounds good", REQUEST),

        # ---- continuation ----
        ("scalability basics", REQUEST),
        ("load balancing", REQUEST),

        # ---- hostility boundary ----
        ("this is annoying", REQUEST),
        ("you are dumb", HOSTILE),
        ("go to hell", HOSTILE),
    ]

    for index, (text, expected_label) in enumerate(CHAT):
        vec = embedder.encode(text)
        pred = clf.predict(vec)
        dwell_label = dwell.apply(pred.label, pred.activation)
        final_label = decide(
            dwell_label,
            pred.confidence,
            confidence_threshold=config.confidence_threshold,
        )
        logger.info(
            "%02d | %r → %s (conf=%.2f, act=%.2f)",
            index,
            text,
            final_label,
            pred.confidence,
            pred.activation,
        )
        assert final_label == expected_label
