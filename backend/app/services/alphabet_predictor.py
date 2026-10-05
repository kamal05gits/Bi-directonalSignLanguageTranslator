"""Lazy, thread-safe Keras alphabet inference."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .keras_classifier import LazyKerasClassifier, Prediction

__all__ = ["AlphabetPredictor", "Prediction"]


class AlphabetPredictor(LazyKerasClassifier):
    """Load the 64x64 RGB classifier on first use and return ranked labels."""

    name = "alphabet"

    def predict(self, image: Image.Image, top_k: int = 3) -> list[Prediction]:
        # The model contains its own 1/255 Rescaling layer; preserve pixel values.
        resized = image.resize((64, 64), Image.Resampling.LANCZOS)
        batch = np.asarray(resized, dtype=np.float32)[None, ...]
        return self._predict_array(batch, top_k)
