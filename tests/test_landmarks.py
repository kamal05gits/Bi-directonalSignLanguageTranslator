import numpy as np
from app.vision.hand_detector import HandDetectionResult
from app.vision.landmarks import FEATURE_DIM, extract_landmark_features


def _hand(offset: float = 0.0):
    return [(offset + i / 20.0, offset + i / 30.0, 0.01 * i) for i in range(21)]


def test_landmarks_are_fixed_size_and_wrist_relative():
    result = HandDetectionResult([_hand()], ["left"], None)
    features = extract_landmark_features(result)
    assert features.shape == (FEATURE_DIM,)
    assert np.allclose(features[:3], 0.0)
    assert np.allclose(features[63:], 0.0)


def test_two_hands_are_kept_in_stable_slots():
    result = HandDetectionResult([_hand(), _hand(0.5)], ["right", "left"], None)
    features = extract_landmark_features(result)
    assert not np.allclose(features[:63], 0.0)
    assert not np.allclose(features[63:], 0.0)
