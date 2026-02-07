# tests/test_hostile_soft_recovery.py

from core.classifier import Classifier
from core.embedder import Embedder
from core.dwell import LabelDwell
from core.decision import decide
from core.types import REQUEST, HOSTILE


def test_soft_hostile_recovery_flow() -> None:
    embedder = Embedder()
    clf = Classifier("models/lr.joblib")
    dwell = LabelDwell()

    user = "hostile_recovery_user"

    # -------------------------
    # Step 1: establish intent
    # -------------------------
    strong = "need help with aws deployment"
    vec_s = embedder.encode(strong)

    label1, conf1, act1 = clf.predict(vec_s, user)
    label1 = decide(dwell.apply(user, label1, act1), conf1)

    print(f"[intent] {label1=} {conf1=:.2f}")
    assert label1 == REQUEST

    # -------------------------
    # Step 2: hostile interruption
    # -------------------------
    hostile = "what the hell"
    vec_h = embedder.encode(hostile)

    label2, conf2, act2 = clf.predict(vec_h, user)
    label2 = decide(dwell.apply(user, label2, act2), conf2)

    print(f"[hostile] {label2=} {conf2=:.2f}")
    assert label2 == HOSTILE

    # -------------------------
    # Step 3: calm noise (still suppressed)
    # -------------------------
    calm = "cool"
    vec_c = embedder.encode(calm)

    label3, conf3, act3 = clf.predict(vec_c, user)
    label3 = decide(dwell.apply(user, label3, act3), conf3)

    print(f"[cooldown-1] {label3=} {conf3=:.2f}")
    assert label3 == HOSTILE  # still suppressed

    # -------------------------
    # Step 4: another calm turn (cooldown exhausted)
    # -------------------------
    calm2 = "okay"
    vec_c2 = embedder.encode(calm2)

    label4, conf4, act4 = clf.predict(vec_c2, user)
    label4 = decide(dwell.apply(user, label4, act4), conf4)

    print(f"[cooldown-2] {label4=} {conf4=:.2f}")
    assert label4 == HOSTILE  # last suppressed turn

    # -------------------------
    # Step 5: productive intent recovers
    # -------------------------
    label5, conf5, act5 = clf.predict(vec_s, user)
    label5 = decide(dwell.apply(user, label5, act5), conf5)

    print(f"[recovered] {label5=} {conf5=:.2f}")
    assert label5 == REQUEST
