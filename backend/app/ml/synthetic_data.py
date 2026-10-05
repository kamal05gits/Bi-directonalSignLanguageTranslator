"""Deterministic placeholder datasets for the Random Forest training scripts.

None of the three recognition models ships a raw training dataset in this
repository (no labeled photos, no captured hand-landmark sequences, no CISLR
video clips) - only the pre-trained model artifacts did, previously as Keras
networks and now as Random Forests. To keep ``/api/predict/*`` fully working
out of the box, each ``train_*_rf.py`` script falls back to a synthetic
dataset generated here when no real data is supplied on the command line.

The synthetic data is **not** real sign-language data. It draws one Gaussian
cluster per label in feature space so a Random Forest can learn a clean
decision boundary end-to-end (proving out the training -> joblib -> API
pipeline), but it will not recognize real photos, hands, or videos. Replace
it with real data (``--images-dir``, ``--landmarks-csv``, ``--features-csv``)
and retrain for actual accuracy.
"""

from __future__ import annotations

import numpy as np


def synthetic_feature_dataset(
    n_classes: int,
    n_features: int,
    samples_per_class: int = 140,
    seed: int = 42,
    low: float = 0.0,
    high: float = 1.0,
    noise: float = 0.06,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` with one well-separated Gaussian cluster per class.

    ``y`` holds integer class indices in ``[0, n_classes)`` - the caller is
    responsible for mapping those indices back to label strings (by position
    in the project's existing ``*_labels.json`` files, so prediction code
    does not need to change).
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(low=low, high=high, size=(n_classes, n_features))
    features, targets = [], []
    for class_index in range(n_classes):
        samples = centers[class_index] + rng.normal(scale=noise, size=(samples_per_class, n_features))
        features.append(np.clip(samples, low, high))
        targets.extend([class_index] * samples_per_class)
    X = np.vstack(features).astype(np.float32)
    y = np.asarray(targets, dtype=np.int64)
    return X, y
