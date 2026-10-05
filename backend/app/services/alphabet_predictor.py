"""Lazy, thread-safe Keras alphabet inference."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


@dataclass
class _Result:
    label: str
    confidence: float


class AlphabetPredictor:
    """Load the 64x64 RGB classifier on first use and return ranked labels."""

    def __init__(self, model_path: Path, labels_path: Path) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self._model: Any = None
        self._lock = threading.Lock()
        self.labels = self._read_labels()

    def _read_labels(self) -> list[str]:
        if not self.labels_path.is_file():
            return []
        labels = json.loads(self.labels_path.read_text(encoding="utf-8"))
        if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
            raise ValueError("Alphabet labels must be a JSON string array.")
        return labels

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                if not self.model_path.is_file():
                    raise RuntimeError(f"Alphabet model is missing: {self.model_path}")
                if not self.labels:
                    raise RuntimeError(f"Alphabet labels are missing: {self.labels_path}")
                try:
                    import tensorflow as tf

                    self._model = tf.keras.models.load_model(self.model_path, compile=False)
                except Exception as exc:  # TensorFlow emits several loader exception types.
                    raise RuntimeError(f"Could not load the alphabet model: {exc}") from exc
        return self._model

    def predict(self, image: Image.Image, top_k: int = 3) -> list[Prediction]:
        model = self._load()
        # The model contains its own 1/255 Rescaling layer; preserve pixel values.
        resized = image.resize((64, 64), Image.Resampling.LANCZOS)
        batch = np.asarray(resized, dtype=np.float32)[None, ...]
        with self._lock:
            probabilities = np.asarray(model.predict(batch, verbose=0))[0]

        if probabilities.ndim != 1 or len(probabilities) != len(self.labels):
            raise RuntimeError("Model output does not match the configured labels.")
        indices = np.argsort(probabilities)[::-1][: max(1, top_k)]
        return [
            Prediction(label=self.labels[int(index)], confidence=float(np.clip(probabilities[index], 0, 1)))
            for index in indices
        ]
