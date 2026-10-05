"""Shared lazy-loading helpers for the bundled Random Forest classifiers.

All three recognition models (alphabet, fingerspelling, word) are
``sklearn.ensemble.RandomForestClassifier`` instances serialized with
``joblib``. They are trained so that the class index lines up with the
position of each label in the matching ``*_labels.json`` file (index ``i``
of ``model.classes_`` always corresponds to ``labels[i]``), so ranking a
prediction never needs to look at ``model.classes_`` at inference time.

scikit-learn is imported lazily so health checks stay fast and the process
never pays the import cost for a model that is never used.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from .model_base import Prediction, read_labels

__all__ = ["LazyRandomForestClassifier", "Prediction", "read_labels"]


class LazyRandomForestClassifier:
    """Load a Random Forest classifier on first use and rank its top-k labels."""

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
                    import joblib

                    model = joblib.load(self.model_path)
                except Exception as exc:  # joblib/pickle raise several loader exception types.
                    raise RuntimeError(f"Could not load the {self.name} model: {exc}") from exc
                n_classes = getattr(model, "n_classes_", None)
                if n_classes is not None and n_classes != len(self.labels):
                    raise RuntimeError(
                        f"{self.name.capitalize()} model has {n_classes} classes but "
                        f"{len(self.labels)} labels are configured."
                    )
                self._model = model
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
            probabilities = np.asarray(model.predict_proba(batch))[0]
        return self._rank(probabilities, top_k)
