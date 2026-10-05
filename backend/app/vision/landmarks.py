import numpy as np
from .hand_detector import HandDetectionResult

FEATURE_DIM = 126

def extract_landmark_features(result: HandDetectionResult) -> np.ndarray:
    output = np.zeros(FEATURE_DIM, dtype=np.float32)
    for hand, side in zip(result.hands, result.handedness):
        if len(hand) != 21: continue
        points = np.asarray(hand, dtype=np.float32)
        points -= points[0]
        slot = 0 if side.lower() == "left" else 63
        output[slot:slot + 63] = points.reshape(-1)
    return output
