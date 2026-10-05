"""Shared helpers for the bundled scikit-learn classifiers.

Historically these models were Keras networks; the project now ships three
``RandomForestClassifier`` models (see ``random_forest_classifier.py``) that
share the same label-file format and ranked-prediction contract, so the
common pieces (the ``Prediction`` result type and the labels-file reader)
live here instead of being duplicated per model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Prediction:
    label: str
    confidence: float


def read_labels(labels_path: Path, name: str) -> list[str]:
    """Read a JSON string-array labels file, returning [] when it is absent."""
    if not labels_path.is_file():
        return []
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
        raise ValueError(f"{name} labels must be a JSON string array.")
    return labels
