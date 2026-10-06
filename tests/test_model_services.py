"""Unit tests for the three model services.

These tests avoid TensorFlow, PyTorch, and OpenCV entirely so they run in the
minimal CI environment (pytest + numpy only).
"""

import json

import numpy as np
import pytest
from app.services.fingerspelling_predictor import FEATURE_DIM, FingerspellingPredictor
from app.services.keras_classifier import LazyKerasClassifier, read_labels
from app.services.word_predictor import (
    IMAGE_SIZE,
    NUM_INPUT_FRAMES,
    WordPredictor,
    is_lfs_pointer,
)


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


class _FakeCapture:
    """Minimal stand-in for ``cv2.VideoCapture`` that yields numbered frames."""

    def __init__(self, total, tracker):
        self._total = total
        self._position = 0
        self._tracker = tracker
        self.released = False

    def isOpened(self):  # noqa: N802 - matches the OpenCV API
        return self._total > 0

    def grab(self):
        if self._position >= self._total:
            return False
        self._position += 1
        return True

    def read(self):
        if self._position >= self._total:
            return False, None
        index = self._position
        self._position += 1
        self._tracker["decoded"].append(index)
        return True, np.full((4, 6, 3), index % 256, dtype=np.uint8)

    def release(self):
        self.released = True


class _FakeCv2:
    COLOR_BGR2RGB = 4
    INTER_LINEAR = 1
    CAP_PROP_FRAME_COUNT = 7

    def __init__(self, total):
        self.total = total
        self.tracker = {"decoded": [], "captures": []}

    def VideoCapture(self, path):  # noqa: N802 - matches the OpenCV API
        capture = _FakeCapture(self.total, self.tracker)
        self.tracker["captures"].append(capture)
        return capture

    def cvtColor(self, frame, code):  # noqa: N802 - matches the OpenCV API
        return frame

    def resize(self, frame, size, interpolation=None):
        width, height = size
        return np.full((height, width, 3), frame.flat[0], dtype=frame.dtype)


def _fake_cv2(monkeypatch, total):
    fake = _FakeCv2(total)
    monkeypatch.setattr(WordPredictor, "_import_cv2", staticmethod(lambda: fake))
    return fake


def test_sample_frames_returns_the_model_input_shape(tmp_path, monkeypatch):
    fake = _fake_cv2(monkeypatch, total=300)
    sampled = WordPredictor._sample_frames(tmp_path / "clip.webm")
    assert sampled.shape == (NUM_INPUT_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3)
    assert sampled.dtype == np.float32
    assert sampled.min() >= -1.0 and sampled.max() <= 1.0
    assert all(capture.released for capture in fake.tracker["captures"])


def test_sample_frames_only_decodes_the_sampled_positions(tmp_path, monkeypatch):
    """A long clip must never be held in memory frame by frame."""
    fake = _fake_cv2(monkeypatch, total=5000)
    WordPredictor._sample_frames(tmp_path / "clip.mp4")
    expected = sorted(set(np.linspace(0, 4999, NUM_INPUT_FRAMES).astype(np.int32).tolist()))
    assert fake.tracker["decoded"] == expected
    assert len(fake.tracker["decoded"]) <= NUM_INPUT_FRAMES


def test_sample_frames_repeats_frames_for_short_clips(tmp_path, monkeypatch):
    fake = _fake_cv2(monkeypatch, total=3)
    sampled = WordPredictor._sample_frames(tmp_path / "clip.webm")
    assert sampled.shape[0] == NUM_INPUT_FRAMES
    assert fake.tracker["decoded"] == [0, 1, 2]


def test_sample_frames_rejects_an_unreadable_video(tmp_path, monkeypatch):
    _fake_cv2(monkeypatch, total=0)
    with pytest.raises(ValueError, match="Could not decode"):
        WordPredictor._sample_frames(tmp_path / "clip.webm")


class _FakePerFrameModel:
    """Returns a scripted distribution per input row, supporting batches."""

    def __init__(self, rows):
        self.rows = np.asarray(rows, dtype=np.float32)

    def predict(self, batch, verbose=0):
        return self.rows[: len(batch)]


def test_fingerspelling_predict_frames_soft_votes_across_samples(tmp_path, monkeypatch):
    predictor = _fingerspelling(tmp_path)
    # Frame 1 spikes on "b", frames 2-3 consistently favour "a": the average
    # must rank "a" first even though one frame disagreed.
    monkeypatch.setattr(
        predictor, "_load", lambda: _FakePerFrameModel([[0.1, 0.9], [0.8, 0.2], [0.8, 0.2]])
    )
    frames = [[0.1] * FEATURE_DIM, [0.2] * FEATURE_DIM, [0.3] * FEATURE_DIM]
    result = predictor.predict_frames(frames, top_k=2)
    assert result[0].label == "a"
    assert result[0].confidence == pytest.approx((0.1 + 0.8 + 0.8) / 3)


def test_fingerspelling_predict_frames_drops_empty_frames(tmp_path, monkeypatch):
    predictor = _fingerspelling(tmp_path)
    monkeypatch.setattr(predictor, "_load", lambda: _FakePerFrameModel([[0.2, 0.8]]))
    frames = [[0.0] * FEATURE_DIM, [0.1] * FEATURE_DIM]
    result = predictor.predict_frames(frames, top_k=2)
    assert result[0].label == "b"


def test_fingerspelling_predict_frames_rejects_all_empty(tmp_path):
    predictor = _fingerspelling(tmp_path)
    with pytest.raises(ValueError, match="No hand"):
        predictor.predict_frames([[0.0] * FEATURE_DIM, [0.0] * FEATURE_DIM])


# ------------------------------------------------------- calibration & mirror


def test_calibration_softens_saturated_distributions(tmp_path, monkeypatch):
    predictor = FingerspellingPredictor(
        tmp_path / "model.keras", _write_labels(tmp_path), temperature=2.0
    )
    monkeypatch.setattr(predictor, "_load", lambda: _FakePerFrameModel([[0.9, 0.1]]))
    result = predictor.predict([0.1] * FEATURE_DIM, top_k=2)
    # p^(1/2) renormalized: 0.9487 / (0.9487 + 0.3162) = 0.75
    assert result[0].label == "a"
    assert result[0].confidence == pytest.approx(0.75, abs=1e-3)
    assert result[1].confidence == pytest.approx(0.25, abs=1e-3)


def test_temperature_one_leaves_distribution_unchanged(tmp_path, monkeypatch):
    predictor = _fingerspelling(tmp_path)
    monkeypatch.setattr(predictor, "_load", lambda: _FakePerFrameModel([[0.9, 0.1]]))
    result = predictor.predict([0.1] * FEATURE_DIM, top_k=2)
    assert result[0].confidence == pytest.approx(0.9)


def test_invalid_temperature_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="temperature"):
        FingerspellingPredictor(tmp_path / "model.keras", _write_labels(tmp_path), temperature=0)


def test_mirror_hands_swaps_slots_and_negates_x():
    from app.services.fingerspelling_predictor import HAND_DIM, mirror_hands

    vector = np.zeros(FEATURE_DIM, dtype=np.float32)
    vector[0:3] = [0.5, 0.25, -0.125]  # left-hand point (x, y, z)
    mirrored = mirror_hands(vector)
    assert mirrored[HAND_DIM:HAND_DIM + 3] == pytest.approx([-0.5, 0.25, -0.125])
    assert not mirrored[:HAND_DIM].any()


class _SlotSensitiveModel:
    """Scores rows with a populated right-hand slot far higher than left-only rows."""

    def predict(self, batch, verbose=0):
        rows = []
        for row in batch:
            rows.append([0.05, 0.95] if np.any(row[63:]) else [0.6, 0.4])
        return np.asarray(rows, dtype=np.float32)


def test_mirror_tta_rescues_wrong_slot_hand(tmp_path, monkeypatch):
    predictor = FingerspellingPredictor(
        tmp_path / "model.keras", _write_labels(tmp_path), mirror_tta=True
    )
    monkeypatch.setattr(predictor, "_load", lambda: _SlotSensitiveModel())
    vector = [0.0] * FEATURE_DIM
    vector[0] = 0.3  # hand landed in the slot the model was never trained on
    result = predictor.predict(vector, top_k=2)
    assert result[0].label == "b"
    assert result[0].confidence == pytest.approx(0.95)


def test_mirror_tta_keeps_original_when_not_clearly_better(tmp_path, monkeypatch):
    predictor = FingerspellingPredictor(
        tmp_path / "model.keras", _write_labels(tmp_path), mirror_tta=True, mirror_margin=1.25
    )

    class _NearTie:
        def predict(self, batch, verbose=0):
            # Mirrored rows score slightly higher, but inside the margin.
            rows = [[0.7, 0.3] if not np.any(row[63:]) else [0.25, 0.75] for row in batch]
            return np.asarray(rows, dtype=np.float32)

    monkeypatch.setattr(predictor, "_load", lambda: _NearTie())
    vector = [0.0] * FEATURE_DIM
    vector[0] = 0.3
    result = predictor.predict(vector, top_k=2)
    assert result[0].label == "a"  # 0.75 < 0.7 * 1.25, so the original stands
