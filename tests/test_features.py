import numpy as np
import pytest

from app.features.normalization import FeatureScaler
from app.features.sequence_buffer import SequenceBuffer


def test_sequence_buffer_is_bounded():
    buffer = SequenceBuffer(3, 4)
    for value in range(5):
        buffer.add(np.full(4, value, dtype=np.float32))
    assert buffer.ready
    assert buffer.as_array()[:, 0].tolist() == [2, 3, 4]


def test_sequence_buffer_rejects_wrong_dimensions():
    buffer = SequenceBuffer(3, 4)
    with pytest.raises(ValueError):
        buffer.add(np.zeros(3))


def test_scaler_round_trip_shape():
    values = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    scaler = FeatureScaler().fit(values)
    transformed = scaler.transform(values)
    assert transformed.shape == values.shape
    assert np.allclose(transformed.reshape(-1, 4).mean(axis=0), 0.0)
