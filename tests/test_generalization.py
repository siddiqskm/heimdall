# tests/test_generalization.py
#
# Tests that the classifier generalizes to novel phrasings via lowered thresholds,
# diverse semantic anchors, and the soft nearest-neighbor fallback (when LR
# confidence is low). Ensures we do not over-trigger (e.g. REQUEST stays REQUEST).

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


def _run_pipeline(
    config: HeimdallConfig,
    embedder: Embedder,
    clf: Classifier,
    dwell: LabelDwell,
    text: str,
) -> Label:
    vector = embedder.encode(text)
    pred = clf.predict(vector)
    dwell_label = dwell.apply(pred.label, pred.activation)
    return decide(
        dwell_label,
        pred.confidence,
        confidence_threshold=config.confidence_threshold,
    )


def test_novel_phrasing_topic_reset(tmp_path: Path) -> None:
    """
    Novel phrasings NOT in training data should still be classified as TOPIC_RESET
    via prototype similarity (lowered offline_proto_threshold), score override
    (lowered reset_threshold), or soft fallback (when LR is uncertain).
    """
    config = HeimdallConfig(state_dir=tmp_path / ".heimdall")
    embedder = Embedder()
    clf = Classifier(config=config)
    dwell = LabelDwell(config=config, chat_id=clf.chat_id)

    # These phrases are not literal entries in DATA; they should generalize.
    # (Omit borderline phrasings that may reasonably be REQUEST, e.g. "i want to talk about something different")
    novel_reset_phrases: list[str] = [
        "nah not feeling it, lets do something else",
        "no not in mood, lets switch to something else",
        "can we change the subject",
    ]

    for text in novel_reset_phrases:
        final_label = _run_pipeline(config, embedder, clf, dwell, text)
        logger.info("%r → %s", text, final_label)
        assert final_label == TOPIC_RESET, f"Expected TOPIC_RESET for {text!r}, got {final_label}"


def test_no_false_positive_reset(tmp_path: Path) -> None:
    """
    Clearly non-reset messages must remain REQUEST (or SILENT); lowered
    thresholds must not cause false TOPIC_RESET.
    """
    config = HeimdallConfig(state_dir=tmp_path / ".heimdall")
    embedder = Embedder()
    clf = Classifier(config=config)
    dwell = LabelDwell(config=config, chat_id=clf.chat_id)

    should_stay_request: list[str] = [
        "tell me more about auth",
        "lets discuss oauth",
        "go on",
        "continue please",
        "expand on that",
    ]

    for text in should_stay_request:
        final_label = _run_pipeline(config, embedder, clf, dwell, text)
        logger.info("%r → %s", text, final_label)
        assert final_label == REQUEST, f"Expected REQUEST for {text!r}, got {final_label}"


def test_soft_fallback_low_confidence(tmp_path: Path) -> None:
    """
    When LR confidence is low (< lr_low_confidence_threshold) and the input
    is semantically close to a non-LR label (e.g. TOPIC_RESET), the soft
    nearest-neighbor fallback or score override should yield the correct label.
    """
    config = HeimdallConfig(state_dir=tmp_path / ".heimdall")
    embedder = Embedder()
    clf = Classifier(config=config)
    dwell = LabelDwell(config=config, chat_id=clf.chat_id)

    # Phrase that previously got REQUEST@low confidence; should now get TOPIC_RESET.
    text = "no not in mood, lets switch to something else"
    vector = embedder.encode(text)
    pred = clf.predict(vector)
    dwell_label = dwell.apply(pred.label, pred.activation)
    final_label = decide(
        dwell_label,
        pred.confidence,
        confidence_threshold=config.confidence_threshold,
    )

    logger.info(
        "%r → label=%s final=%s conf=%.2f reset_score=%.2f",
        text,
        pred.label,
        final_label,
        pred.confidence,
        pred.reset_score,
    )
    assert final_label == TOPIC_RESET, (
        f"Expected TOPIC_RESET (soft fallback or thresholds); got {final_label} "
        f"(label={pred.label}, conf={pred.confidence:.2f})"
    )


def test_hostile_and_silent_unchanged(tmp_path: Path) -> None:
    """Hostile and silent exemplars still classify correctly after generalization changes."""
    config = HeimdallConfig(state_dir=tmp_path / ".heimdall")
    embedder = Embedder()
    clf = Classifier(config=config)
    dwell = LabelDwell(config=config, chat_id=clf.chat_id)

    cases: list[tuple[str, Label]] = [
        ("go to hell", HOSTILE),
        ("shut up", HOSTILE),
        ("", SILENT),
        ("ok", SILENT),
        ("cool", SILENT),
    ]

    for text, expected in cases:
        final_label = _run_pipeline(config, embedder, clf, dwell, text)
        logger.info("%r → %s (expected %s)", text, final_label, expected)
        assert final_label == expected, f"Expected {expected} for {text!r}, got {final_label}"
