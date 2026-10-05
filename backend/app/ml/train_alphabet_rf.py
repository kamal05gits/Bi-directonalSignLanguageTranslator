"""Train the alphabet Random Forest classifier.

Usage::

    python -m app.ml.train_alphabet_rf
    python -m app.ml.train_alphabet_rf --images-dir /path/to/dataset

With ``--images-dir``, the dataset must have one subfolder per label
containing JPEG/PNG/WebP photos (the common layout for Kaggle-style ISL/ASL
alphabet datasets), e.g.::

    dataset/
      a/ img001.jpg img002.jpg ...
      b/ ...
      ...

Without ``--images-dir``, a synthetic placeholder dataset is generated
instead so the training -> joblib -> API pipeline keeps working out of the
box. It is **not** real photos and will not recognize real hands - see
``app.ml.synthetic_data`` - so plug in a real dataset with ``--images-dir``
before relying on this for real recognition accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .rf_training import report_accuracy, save_model, split_dataset, train_random_forest
from .synthetic_data import synthetic_feature_dataset

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABELS = ROOT / "backend/models/alphabet/alphabet_labels.json"
DEFAULT_OUTPUT = ROOT / "backend/models/alphabet/isl_alphabet_model_rf.joblib"
IMAGE_SIZE = 32  # Must match app.services.alphabet_predictor.IMAGE_SIZE.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _load_labels(path: Path) -> list[str]:
    labels = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
        raise ValueError(f"{path} must contain a JSON array of label strings.")
    return labels


def _load_images_dataset(images_dir: Path, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    label_index = {label: index for index, label in enumerate(labels)}
    features: list[np.ndarray] = []
    targets: list[int] = []
    for class_dir in sorted(Path(images_dir).iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name.strip().lower()
        if label not in label_index:
            print(f"Skipping unexpected label directory: {class_dir.name}")
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            with Image.open(image_path) as image:
                resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
                features.append(np.asarray(resized, dtype=np.float32).reshape(-1) / 255.0)
            targets.append(label_index[label])
    if not features:
        raise ValueError(f"No images found under {images_dir}")
    return np.stack(features), np.asarray(targets, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", type=Path, default=None, help="Directory of label subfolders with real photos.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--samples-per-class", type=int, default=160, help="Only used for the synthetic fallback dataset."
    )
    args = parser.parse_args()

    labels = _load_labels(args.labels)
    if args.images_dir is not None:
        print(f"Loading real photos from {args.images_dir} ...")
        X, y = _load_images_dataset(args.images_dir, labels)
    else:
        print(
            "No --images-dir given: generating a synthetic placeholder dataset "
            "(see app.ml.synthetic_data) so the API has a working model."
        )
        X, y = synthetic_feature_dataset(
            n_classes=len(labels),
            n_features=IMAGE_SIZE * IMAGE_SIZE * 3,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(X, y, seed=args.seed)
    model = train_random_forest(
        X_train, y_train, n_estimators=args.n_estimators, max_depth=args.max_depth, seed=args.seed
    )

    print("Alphabet Random Forest:")
    report_accuracy("train", model, X_train, y_train)
    report_accuracy("validation", model, X_val, y_val)
    report_accuracy("test", model, X_test, y_test)
    save_model(model, args.output)


if __name__ == "__main__":
    main()
