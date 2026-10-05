"""Shared helpers for the three ``train_*_rf.py`` Random Forest scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.dataset_manager import stratified_split


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    """Stratified train/validation/test split, reusing the project's splitter."""
    train_idx, validation_idx, test_idx = stratified_split(
        y, seed=seed, validation_ratio=validation_ratio, test_ratio=test_ratio
    )
    return (
        (X[train_idx], y[train_idx]),
        (X[validation_idx], y[validation_idx]),
        (X[test_idx], y[test_idx]),
    )


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 300,
    max_depth: int | None = None,
    seed: int = 42,
    n_jobs: int = -1,
    **kwargs,
):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=n_jobs,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model


def report_accuracy(name: str, model, X: np.ndarray, y: np.ndarray) -> float | None:
    if len(y) == 0:
        print(f"  {name}: (empty split, skipped)")
        return None
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y, model.predict(X))
    print(f"  {name}: accuracy={accuracy:.3f} ({len(y)} samples)")
    return accuracy


def save_model(model, output_path: Path, compress: int = 3) -> None:
    """Serialize ``model`` with joblib. ``compress`` keeps the file small -
    a few hundred trees over a few thousand features otherwise serializes to
    tens of megabytes uncompressed."""
    import joblib

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path, compress=compress)
    size_mb = output_path.stat().st_size / 1e6
    print(f"Saved Random Forest model -> {output_path} ({size_mb:.2f} MB)")
