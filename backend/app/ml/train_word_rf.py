"""Train the CISLR word Random Forest classifier.

Usage::

    python -m app.ml.train_word_rf
    python -m app.ml.train_word_rf --features-csv /path/to/pooled_features.csv

``--features-csv`` must have a ``label`` column plus 2048 numeric columns:
the mean and standard deviation (in that order, each 1024-wide) of a clip's
normalized I3D feature sequence, exactly what
``app.services.word_predictor.pool_features`` computes from a real video
clip. Run the existing I3D feature-extraction pipeline
(``app.services.word_predictor.WordPredictor._extract_features`` plus the
``CISLR_NORMALIZATION.npz`` stats) over real CISLR clips, pool each clip with
``pool_features``, and write the rows to this CSV to train on real data.

Without ``--features-csv``, a synthetic placeholder dataset is generated
instead so the training -> joblib -> API pipeline keeps working out of the
box. It is **not** real video data - see ``app.ml.synthetic_data`` - so
supply real pooled features before relying on this for real recognition
accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .rf_training import report_accuracy, save_model, split_dataset, train_random_forest
from .synthetic_data import synthetic_feature_dataset

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABELS = ROOT / "backend/models/cislr/CISLR_LABELS.json"
DEFAULT_OUTPUT = ROOT / "backend/models/cislr/CISLR_MODEL_rf.joblib"
I3D_FEATURE_DIM = 1024
POOLED_FEATURE_DIM = I3D_FEATURE_DIM * 2  # mean + std, see word_predictor.pool_features.


def _load_labels(path: Path) -> list[str]:
    labels = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
        raise ValueError(f"{path} must contain a JSON array of label strings.")
    return labels


def _load_features_csv(csv_path: Path, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    label_index = {label: index for index, label in enumerate(labels)}
    features: list[list[float]] = []
    targets: list[int] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = row["label"].strip().lower()
            if label not in label_index:
                print(f"Skipping unknown label: {label}")
                continue
            values = [float(value) for key, value in row.items() if key != "label"]
            if len(values) != POOLED_FEATURE_DIM:
                raise ValueError(f"Expected {POOLED_FEATURE_DIM} pooled feature columns, got {len(values)}.")
            features.append(values)
            targets.append(label_index[label])
    if not features:
        raise ValueError(f"No rows found in {csv_path}")
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-csv", type=Path, default=None, help="label,value x2048 CSV of pooled I3D features.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--samples-per-class", type=int, default=60, help="Only used for the synthetic fallback dataset."
    )
    args = parser.parse_args()

    labels = _load_labels(args.labels)
    if args.features_csv is not None:
        print(f"Loading real pooled I3D features from {args.features_csv} ...")
        X, y = _load_features_csv(args.features_csv, labels)
    else:
        print(
            "No --features-csv given: generating a synthetic placeholder dataset "
            "(see app.ml.synthetic_data) so the API has a working model."
        )
        X, y = synthetic_feature_dataset(
            n_classes=len(labels),
            n_features=POOLED_FEATURE_DIM,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
            low=-1.0,
            high=1.5,
            noise=0.25,
        )

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(X, y, seed=args.seed)
    model = train_random_forest(
        X_train, y_train, n_estimators=args.n_estimators, max_depth=args.max_depth, seed=args.seed
    )

    print("Word (CISLR) Random Forest:")
    report_accuracy("train", model, X_train, y_train)
    report_accuracy("validation", model, X_val, y_val)
    report_accuracy("test", model, X_test, y_test)
    save_model(model, args.output)


if __name__ == "__main__":
    main()
