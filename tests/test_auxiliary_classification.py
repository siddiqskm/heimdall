# tests/test_auxiliary_classification.py

from pathlib import Path

from heimdall import Classifier, Embedder, HeimdallConfig


def test_auxiliary_classification_basic(tmp_path: Path):
    """Test basic auxiliary classification."""
    config = HeimdallConfig(state_dir=tmp_path)

    # Define test corpus
    auxiliary_corpora = {
        "mood": {
            "happy": ["User is excited", "User expresses joy"],
            "sad": ["User is disappointed", "User expresses sadness"],
        }
    }

    clf = Classifier(config=config, auxiliary_corpora=auxiliary_corpora)
    embedder = Embedder()

    # Test happy classification
    vec = embedder.encode("I'm so excited about this!")
    pred = clf.predict(vec)

    assert pred.auxiliary is not None
    assert "mood" in pred.auxiliary
    mood, conf = pred.auxiliary["mood"]
    assert mood == "happy"
    assert conf > 0.3  # Lower threshold for auxiliary classification


def test_auxiliary_classification_multiple_namespaces(tmp_path: Path):
    """Test multiple auxiliary namespaces."""
    config = HeimdallConfig(state_dir=tmp_path)

    auxiliary_corpora = {
        "mood": {
            "happy": ["User is excited"],
            "sad": ["User is sad"],
        },
        "urgency": {
            "high": ["User needs this now"],
            "low": ["User is just browsing"],
        }
    }

    clf = Classifier(config=config, auxiliary_corpora=auxiliary_corpora)
    embedder = Embedder()

    vec = embedder.encode("I need this urgently!")
    pred = clf.predict(vec)

    assert pred.auxiliary is not None
    assert "mood" in pred.auxiliary
    assert "urgency" in pred.auxiliary


def test_auxiliary_classification_none_when_not_provided(tmp_path: Path):
    """Test that auxiliary is None when no corpus provided."""
    config = HeimdallConfig(state_dir=tmp_path)
    clf = Classifier(config=config)
    embedder = Embedder()

    vec = embedder.encode("Hello")
    pred = clf.predict(vec)

    assert pred.auxiliary is None


def test_backward_compatibility_tuple_unpacking(tmp_path: Path):
    """Test that tuple unpacking still works."""
    config = HeimdallConfig(state_dir=tmp_path)
    clf = Classifier(config=config)
    embedder = Embedder()

    vec = embedder.encode("Hello")
    pred = clf.predict(vec)

    # Should still support: label, conf, act = pred
    label, conf, act = pred
    assert label is not None
    assert isinstance(conf, float)
    assert isinstance(act, float)


def test_auxiliary_with_cog_tool_intent_corpus(tmp_path: Path):
    """Test with Cog-like tool intent corpus."""
    config = HeimdallConfig(state_dir=tmp_path)

    tool_intent_corpus = {
        "chat": [
            "User continues conversation naturally",
            "User asks a question about current topic",
        ],
        "search": [
            "User wants to find what was said",
            "User asks 'what did we discuss'",
        ],
        "summarize": [
            "User says 'summarize this chat'",
            "User requests 'create a summary'",
        ],
    }

    clf = Classifier(
        config=config,
        auxiliary_corpora={"tool_intent": tool_intent_corpus}
    )
    embedder = Embedder()

    # Test chat intent
    vec = embedder.encode("I also want to add hybrid search support")
    pred = clf.predict(vec, text="I also want to add hybrid search support")

    assert pred.auxiliary is not None
    assert "tool_intent" in pred.auxiliary
    intent, conf = pred.auxiliary["tool_intent"]
    # Could match either "chat" or "search" due to word "search" in text
    assert intent in ("chat", "search")
    assert conf > 0.1  # Low threshold - just verify best match

    # Test summarize intent
    vec = embedder.encode("Please summarize our discussion")
    pred = clf.predict(vec, text="Please summarize our discussion")

    assert pred.auxiliary is not None
    assert "tool_intent" in pred.auxiliary
    intent, conf = pred.auxiliary["tool_intent"]
    assert intent == "summarize"
    assert conf > 0.1  # Very low threshold - just verify best match

    # Test search intent
    vec = embedder.encode("What did we decide about the database?")
    pred = clf.predict(vec, text="What did we decide about the database?")

    assert pred.auxiliary is not None
    assert "tool_intent" in pred.auxiliary
    intent, conf = pred.auxiliary["tool_intent"]
    assert intent in ("search", "chat")  # Could be either


def test_auxiliary_populated_on_prototype_early_return(tmp_path: Path):
    """Auxiliary results must be present even when a prototype match short-circuits the LR path."""
    config = HeimdallConfig(state_dir=tmp_path)
    auxiliary_corpora = {
        "mood": {
            "happy": ["User is excited", "User expresses joy"],
            "sad": ["User is disappointed", "User expresses sadness"],
        }
    }

    embedder = Embedder()
    clf = Classifier(config=config, auxiliary_corpora=auxiliary_corpora, embedder=embedder)

    # Seed a session prototype so the next identical call hits the early-return path.
    from heimdall.core.types import REQUEST
    vec = embedder.encode("I'm so excited about this!")
    clf.session_prototypes.add(REQUEST, vec)

    pred = clf.predict(vec)

    # Core label came from prototype, not LR — but auxiliary must still be populated.
    assert pred.auxiliary is not None, (
        "auxiliary was None on a prototype early-return; _compute_auxiliary must run before early returns"
    )
    assert "mood" in pred.auxiliary


def test_auxiliary_embedder_injection(tmp_path: Path):
    """Passing an existing Embedder avoids a redundant model load."""
    config = HeimdallConfig(state_dir=tmp_path)
    embedder = Embedder()

    auxiliary_corpora = {
        "mood": {
            "happy": ["User is excited"],
            "sad": ["User is sad"],
        }
    }

    # Embedder is injected; no second SentenceTransformer should be constructed internally.
    clf = Classifier(config=config, auxiliary_corpora=auxiliary_corpora, embedder=embedder)

    vec = embedder.encode("I'm thrilled!")
    pred = clf.predict(vec)

    assert pred.auxiliary is not None
    assert "mood" in pred.auxiliary


def test_auxiliary_threshold_filters_low_confidence(tmp_path: Path):
    """With a high threshold, a poor-match namespace is excluded from results."""
    config = HeimdallConfig(state_dir=tmp_path)

    # Corpus whose categories are semantically very different from the query below.
    auxiliary_corpora = {
        "topic": {
            "cooking": ["Recipe instructions", "How to bake a cake"],
            "gardening": ["Planting seeds", "Watering flowers"],
        }
    }

    embedder = Embedder()
    # High threshold — an off-topic query should produce no match.
    clf = Classifier(
        config=config,
        auxiliary_corpora=auxiliary_corpora,
        embedder=embedder,
        auxiliary_threshold=0.99,  # Near-perfect similarity required
    )

    vec = embedder.encode("What is the capital of France?")
    pred = clf.predict(vec)

    # No category should clear the very high threshold.
    assert pred.auxiliary is None
