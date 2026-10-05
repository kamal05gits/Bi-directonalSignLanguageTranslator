"""Landmark-vector inference for the fingerspelling Random Forest classifier."""

from __future__ import annotations

import numpy as np

from .random_forest_classifier import LazyRandomForestClassifier, Prediction

# 2 hands x 21 landmarks x (x, y, z); matches app.vision.landmarks.FEATURE_DIM.
FEATURE_DIM = 126

__all__ = ["FingerspellingPredictor", "Prediction", "FEATURE_DIM"]


class FingerspellingPredictor(LazyRandomForestClassifier):
    """Classify a 126-value hand-landmark vector into an ISL alphabet letter.

    The vector layout matches ``app.vision.landmarks.extract_landmark_features``:
    the left hand occupies values 0-62 and the right hand values 63-125, with
    each point stored as wrist-relative ``(x, y, z)`` and invisible hands left
    as zeros. Landmarks are extracted in the browser with MediaPipe, so the
    server needs neither a camera nor MediaPipe itself. A Random Forest
    classifies the vector directly - no neural network is involved.
    """

    name = "fingerspelling"

    def __init__(self, model_path, labels_path, feature_dim: int = FEATURE_DIM) -> None:
        super().__init__(model_path, labels_path)
        self.feature_dim = feature_dim

    def predict(self, landmarks, top_k: int = 3) -> list[Prediction]:
        vector = np.asarray(landmarks, dtype=np.float32).reshape(-1)
        if vector.shape != (self.feature_dim,):
            raise ValueError(
                f"Expected {self.feature_dim} landmark values "
                f"(2 hands x 21 points x 3 coordinates), got {vector.size}."
            )
        if not np.isfinite(vector).all():
            raise ValueError("Landmark values must be finite numbers.")
        if not vector.any():
            raise ValueError("No hand landmarks were detected. Show one hand to the camera and try again.")
        return self._predict_array(vector[None, :], top_k)
