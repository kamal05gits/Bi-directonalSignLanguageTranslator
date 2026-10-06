import numpy as np
from app.ml.predict import LabelStabilizer, PredictionEngine


class FakeModel:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def predict(self, values, verbose=0):
        assert values.shape == (1, 3, 2)
        return self.values[None, :]


def test_low_confidence_is_rejected():
    engine = PredictionEngine(FakeModel([0.6, 0.4]), ["hello", "no"], confidence_threshold=0.75)
    result = engine.predict(np.zeros((3, 2), dtype=np.float32))
    assert result.label == "hello"
    assert result.accepted is False


def test_high_confidence_is_accepted_and_duplicate_is_suppressed():
    engine = PredictionEngine(
        FakeModel([0.95, 0.05]), ["hello", "no"], confidence_threshold=0.75, cooldown_seconds=60
    )
    first = engine.predict(np.zeros((3, 2), dtype=np.float32))
    second = engine.predict(np.zeros((3, 2), dtype=np.float32))
    assert first.accepted
    assert not second.accepted
    assert "Duplicate" in second.reason


def test_label_stabilizer_requires_a_consecutive_hold_before_accepting():
    stabilizer = LabelStabilizer(confidence_threshold=0.7, hold_frames=3, cooldown_seconds=5)
    first = stabilizer.update("a", 0.9, now=0.0)
    second = stabilizer.update("a", 0.9, now=0.1)
    third = stabilizer.update("a", 0.9, now=0.2)
    assert [first.accepted, second.accepted, third.accepted] == [False, False, True]


def test_label_stabilizer_rejects_low_confidence_and_resets_streak():
    stabilizer = LabelStabilizer(confidence_threshold=0.7, hold_frames=2)
    stabilizer.update("a", 0.9, now=0.0)
    low = stabilizer.update("a", 0.3, now=0.1)
    assert not low.accepted
    assert low.reason == "Low confidence"
    # The streak reset on low confidence, so "a" needs its full hold again.
    again = stabilizer.update("a", 0.9, now=0.2)
    assert not again.accepted


def test_label_stabilizer_suppresses_duplicates_within_cooldown_but_not_after():
    stabilizer = LabelStabilizer(confidence_threshold=0.7, hold_frames=1, cooldown_seconds=1.0)
    first = stabilizer.update("a", 0.9, now=0.0)
    duplicate = stabilizer.update("a", 0.9, now=0.2)
    later = stabilizer.update("a", 0.9, now=1.5)
    assert first.accepted
    assert not duplicate.accepted and duplicate.reason == "Duplicate prediction suppressed"
    assert later.accepted


def test_label_stabilizer_accepts_a_different_label_immediately_after_its_own_hold():
    stabilizer = LabelStabilizer(confidence_threshold=0.7, hold_frames=1, cooldown_seconds=5)
    first = stabilizer.update("a", 0.9, now=0.0)
    second = stabilizer.update("b", 0.9, now=0.1)
    assert first.accepted
    assert second.accepted
