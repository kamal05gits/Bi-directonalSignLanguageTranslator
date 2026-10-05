"""Shared lazy-loading helpers for the bundled Keras classifiers."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


def read_labels(labels_path: Path, name: str) -> list[str]:
    """Read a JSON string-array labels file, returning [] when it is absent."""
    if not labels_path.is_file():
        return []
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
        raise ValueError(f"{name} labels must be a JSON string array.")
    return labels


class LazyKerasClassifier:
    """Load a Keras classifier on first use and rank its top-k labels.

    TensorFlow is imported lazily so health checks stay fast and deployments
    that only need a subset of the models do not pay the startup cost twice.
    """

    name = "model"

    def __init__(self, model_path: Path, labels_path: Path) -> None:
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self._model: Any = None
        self._lock = threading.Lock()
        self.labels = read_labels(self.labels_path, self.name)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def available(self) -> bool:
        return self.model_path.is_file() and bool(self.labels)

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                if not self.model_path.is_file():
                    raise RuntimeError(f"{self.name.capitalize()} model is missing: {self.model_path}")
                if not self.labels:
                    raise RuntimeError(f"{self.name.capitalize()} labels are missing: {self.labels_path}")
                try:
                    import tensorflow as tf

                    self._model = tf.keras.models.load_model(self.model_path, compile=False)
                except Exception as exc:  # TensorFlow emits several loader exception types.
                    raise RuntimeError(f"Could not load the {self.name} model: {exc}") from exc
        return self._model

    def _rank(self, probabilities: np.ndarray, top_k: int) -> list[Prediction]:
        probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if probabilities.size != len(self.labels):
            raise RuntimeError(f"{self.name.capitalize()} model output does not match the configured labels.")
        indices = np.argsort(probabilities)[::-1][: max(1, top_k)]
        return [
            Prediction(label=self.labels[int(index)], confidence=float(np.clip(probabilities[index], 0, 1)))
            for index in indices
        ]

    def _predict_array(self, batch: np.ndarray, top_k: int) -> list[Prediction]:
        model = self._load()
        with self._lock:
            probabilities = np.asarray(model.predict(batch, verbose=0))[0]
        return self._rank(probabilities, top_k)
