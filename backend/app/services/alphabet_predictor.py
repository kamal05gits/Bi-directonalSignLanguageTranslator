"""Lazy, thread-safe Random Forest alphabet inference."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .random_forest_classifier import LazyRandomForestClassifier, Prediction

__all__ = ["AlphabetPredictor", "Prediction", "IMAGE_SIZE"]

# Keep the photo small so the flattened pixel vector stays a manageable
# Random Forest input (32x32x3 = 3072 features).
IMAGE_SIZE = 32


class AlphabetPredictor(LazyRandomForestClassifier):
    """Resize a photo to a flat pixel vector and classify it with a Random Forest."""

    name = "alphabet"

    def predict(self, image: Image.Image, top_k: int = 3) -> list[Prediction]:
        resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        batch = (np.asarray(resized, dtype=np.float32) / 255.0).reshape(1, -1)
        return self._predict_array(batch, top_k)
