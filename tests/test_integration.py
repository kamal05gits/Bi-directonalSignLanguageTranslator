import numpy as np

from app.features.extraction import extract_frame_features
from app.features.sequence_buffer import SequenceBuffer
from app.ml.predict import PredictionEngine
from app.vision.hand_detector import HandDetectionResult


class Model:
    def predict(self, values, verbose=0):
        return np.asarray([[0.9, 0.1]], dtype=np.float32)


def test_landmark_to_prediction_pipeline():
    hand = [(i / 20.0, i / 30.0, 0.0) for i in range(21)]
    detection = HandDetectionResult([hand], ["left"], None)
    buffer = SequenceBuffer(2, 126)
    for _ in range(2):
        buffer.add(extract_frame_features(detection))
    engine = PredictionEngine(Model(), ["hello", "no"], confidence_threshold=0.8)
    prediction = engine.predict(buffer.as_array())
    assert prediction.accepted
    assert prediction.label == "hello"
