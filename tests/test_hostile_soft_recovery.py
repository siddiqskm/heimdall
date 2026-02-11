# tests/test_hostile_soft_recovery.py

from pathlib import Path

from heimdall.core.classifier import Classifier
from heimdall.core.config import HeimdallConfig
from heimdall.core.decision import decide
from heimdall.core.dwell import LabelDwell
from heimdall.core.embedder import Embedder
from heimdall.core.types import HOSTILE, REQUEST


def test_soft_hostile_recovery_flow(tmp_path: Path) -> None:
    """
    Ensures that:
    1. Strong intent is established
    2. Hostility overrides
    3. Hostile suppression persists briefly
    4. Productive intent eventually recovers
    """

    config = HeimdallConfig(state_dir=tmp_path / "heimdall_state")

    embedder = Embedder()
    clf = Classifier(config=config)
    dwell = LabelDwell()

    user = "hostile_recovery_user"

    # -------------------------
    # Step 1: establish intent
    # -------------------------
    strong = "need help with aws deployment"
    vec_s = embedder.encode(strong)

    predicted1, conf1, act1 = clf.predict(vec_s, user)
    label1 = decide(dwell.apply(user, predicted1, act1), conf1)

    print(f"[intent] label={label1} conf={conf1:.2f}")
    assert label1 == REQUEST

    # -------------------------
    # Step 2: hostile interruption
    # -------------------------
    hostile = "what the hell"
    vec_h = embedder.encode(hostile)

    predicted2, conf2, act2 = clf.predict(vec_h, user)
    label2 = decide(dwell.apply(user, predicted2, act2), conf2)

    print(f"[hostile] label={label2} conf={conf2:.2f}")
    assert label2 == HOSTILE

    # -------------------------
    # Step 3: calm noise (still suppressed)
    # -------------------------
    calm = "cool"
    vec_c = embedder.encode(calm)

    predicted3, conf3, act3 = clf.predict(vec_c, user)
    label3 = decide(dwell.apply(user, predicted3, act3), conf3)

    print(f"[cooldown-1] label={label3} conf={conf3:.2f}")
    assert label3 == HOSTILE  # still suppressed

    # -------------------------
    # Step 4: another calm turn (cooldown exhausted)
    # -------------------------
    calm2 = "okay"
    vec_c2 = embedder.encode(calm2)

    predicted4, conf4, act4 = clf.predict(vec_c2, user)
    label4 = decide(dwell.apply(user, predicted4, act4), conf4)

    print(f"[cooldown-2] label={label4} conf={conf4:.2f}")
    assert label4 == HOSTILE  # last suppressed turn

    # -------------------------
    # Step 5: productive intent recovers
    # -------------------------
    predicted5, conf5, act5 = clf.predict(vec_s, user)
    label5 = decide(dwell.apply(user, predicted5, act5), conf5)

    print(f"[recovered] label={label5} conf={conf5:.2f}")
    assert label5 == REQUEST
