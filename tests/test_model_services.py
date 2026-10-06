"""Unit tests for the three model services.

These tests avoid TensorFlow, PyTorch, and OpenCV entirely so they run in the
minimal CI environment (pytest + numpy only).
"""

import json

import numpy as np
import pytest
from app.services.fingerspelling_predictor import FEATURE_DIM, FingerspellingPredictor
from app.services.keras_classifier import LazyKerasClassifier, read_labels
from app.services.word_predictor import WordPredictor, is_lfs_pointer


def _write_labels(tmp_path, name="labels.json", values=None):
    path = tmp_path / name
    path.write_text(json.dumps(values if values is not None else ["a", "b"]), encoding="utf-8")
    return path


def test_read_labels_returns_empty_when_missing(tmp_path):
    assert read_labels(tmp_path / "nope.json", "test") == []


def test_read_labels_rejects_non_string_arrays(tmp_path):
    path = _write_labels(tmp_path, values=[1, 2, 3])
    with pytest.raises(ValueError, match="string array"):
        read_labels(path, "test")


def _fingerspelling(tmp_path):
    return FingerspellingPredictor(tmp_path / "model.keras", _write_labels(tmp_path))


def test_fingerspelling_rejects_wrong_vector_length(tmp_path):
    predictor = _fingerspelling(tmp_path)
    with pytest.raises(ValueError, match=str(FEATURE_DIM)):
        predictor.predict([0.0] * 10)


def test_fingerspelling_rejects_non_finite_values(tmp_path):
    predictor = _fingerspelling(tmp_path)
    vector = [0.0] * FEATURE_DIM
    vector[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        predictor.predict(vector)


def test_fingerspelling_rejects_empty_hands(tmp_path):
    predictor = _fingerspelling(tmp_path)
    with pytest.raises(ValueError, match="No hand"):
        predictor.predict([0.0] * FEATURE_DIM)


class _FakeKerasModel:
    def predict(self, batch, verbose=0):
        assert batch.shape == (1, 2)
        return np.asarray([[0.2, 0.8]], dtype=np.float32)


def test_lazy_classifier_ranks_top_k_without_tensorflow(tmp_path, monkeypatch):
    predictor = LazyKerasClassifier(tmp_path / "model.keras", _write_labels(tmp_path))
    monkeypatch.setattr(predictor, "_load", lambda: _FakeKerasModel())
    result = predictor._predict_array(np.zeros((1, 2), dtype=np.float32), top_k=2)
    assert [item.label for item in result] == ["b", "a"]
    assert result[0].confidence == pytest.approx(0.8)


def _word_predictor(tmp_path):
    return WordPredictor(
        model_path=tmp_path / "CISLR_MODEL.keras",
        labels_path=_write_labels(tmp_path, "CISLR_LABELS.json"),
        normalization_path=tmp_path / "CISLR_NORMALIZATION.npz",
        weights_path=tmp_path / "weights.pt",
        i3d_code_dir=tmp_path / "i3d",
    )


def test_word_availability_reports_missing_classifier(tmp_path):
    available, detail = _word_predictor(tmp_path).availability()
    assert not available
    assert "classifier" in detail.lower()


def test_word_availability_detects_lfs_pointer(tmp_path):
    predictor = _word_predictor(tmp_path)
    predictor.model_path.write_bytes(b"keras")
    np.savez(tmp_path / "CISLR_NORMALIZATION.npz", mean=np.zeros(4), std=np.ones(4))
    predictor.weights_path.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 10\n")
    available, detail = predictor.availability()
    assert not available
    assert "git lfs pull" in detail


def test_is_lfs_pointer(tmp_path):
    pointer = tmp_path / "pointer.pt"
    pointer.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    binary = tmp_path / "binary.pt"
    binary.write_bytes(b"PK\x03\x04realdata")
    assert is_lfs_pointer(pointer)
    assert not is_lfs_pointer(binary)
    assert not is_lfs_pointer(tmp_path / "missing.pt")


def test_word_normalization_requires_mean_and_std(tmp_path):
    predictor = _word_predictor(tmp_path)
    np.savez(tmp_path / "CISLR_NORMALIZATION.npz", other=np.zeros(4))
    with pytest.raises(RuntimeError, match="mean.*std"):
        predictor._load_normalization()
