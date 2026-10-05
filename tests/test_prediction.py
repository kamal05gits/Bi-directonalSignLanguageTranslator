import numpy as np

from app.ml.predict import PredictionEngine


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
