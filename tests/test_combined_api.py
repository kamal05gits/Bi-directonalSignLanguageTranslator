"""API tests for /api/predict/combined, using stub predictors (no TensorFlow)."""

from __future__ import annotations

import io
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

import app.main as main  # noqa: E402
from app.main import app  # noqa: E402
from app.services.fingerspelling_predictor import FEATURE_DIM  # noqa: E402
from app.services.keras_classifier import Prediction  # noqa: E402


class StubLetterPredictor:
    labels = ["a", "b", "c"]
    loaded = True
    available = True

    def __init__(self, label: str = "a", confidence: float = 0.9):
        self.ranking = [Prediction(label=label, confidence=confidence), Prediction(label="b", confidence=0.05)]

    def predict(self, value, top_k=3):
        return self.ranking


class StubWordPredictor:
    labels = ["hello", "yes"]
    loaded = True

    def __init__(self, label: str = "hello", confidence: float = 0.8, available: bool = True):
        self.ranking = [Prediction(label=label, confidence=confidence)]
        self._available = available

    def availability(self):
        return self._available, "ready" if self._available else "PyTorch/OpenCV are not installed."

    def predict_video(self, payload, suffix, top_k=5):
        return self.ranking


def _png_file():
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 255, 255)).save(buffer, "PNG")
    buffer.seek(0)
    return "frame.png", buffer, "image/png"


def _video_file():
    return "sign.webm", io.BytesIO(b"fake-webm"), "video/webm"


def _landmarks(value: float = 0.05):
    vector = [value] * FEATURE_DIM
    return json.dumps(vector)


@pytest.fixture(autouse=True)
def stub_predictors(monkeypatch):
    monkeypatch.setattr(main, "alphabet_predictor", StubLetterPredictor("a", 0.9))
    monkeypatch.setattr(main, "fingerspelling_predictor", StubLetterPredictor("a", 0.8))
    monkeypatch.setattr(main, "word_predictor", StubWordPredictor("hello", 0.8))
    yield


@pytest.fixture()
def client():
    return TestClient(app)


def test_combined_with_all_three_inputs_merges_letters_and_keeps_word(client):
    response = client.post(
        "/api/predict/combined",
        files={"image": _png_file(), "video": _video_file()},
        data={"landmarks": _landmarks()},
    )
    assert response.status_code == 200
    data = response.json()
    # Both letter models agree on 'a': soft vote (0.9 + 0.8) / 2 = 0.85,
    # which beats the word candidate's 0.8, so the letter wins.
    assert data["label"] == "a"
    assert data["confidence"] == pytest.approx(0.85)
    assert data["accepted"] is True
    assert data["agreement"] is True
    assert data["word"]["label"] == "hello"
    assert {source["model"] for source in data["sources"]} == {"alphabet", "fingerspelling", "word"}
    assert all(source["ran"] for source in data["sources"])
    assert data["top_predictions"][0]["label"] == "a"


def test_combined_word_candidate_wins_when_it_outscores_letters(client, monkeypatch):
    monkeypatch.setattr(main, "alphabet_predictor", StubLetterPredictor("a", 0.4))
    monkeypatch.setattr(main, "fingerspelling_predictor", StubLetterPredictor("a", 0.4))
    response = client.post(
        "/api/predict/combined",
        files={"image": _png_file(), "video": _video_file()},
        data={"landmarks": _landmarks()},
    )
    data = response.json()
    assert data["label"] == "hello"
    assert data["accepted"] is True
    assert data["word"]["label"] == "hello"


def test_combined_with_landmarks_only(client):
    response = client.post("/api/predict/combined", data={"landmarks": _landmarks()})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "a"
    assert data["agreement"] is None
    ran = {source["model"]: source["ran"] for source in data["sources"]}
    assert ran == {"alphabet": False, "fingerspelling": True, "word": False}


def test_combined_with_image_only(client):
    response = client.post("/api/predict/combined", files={"image": _png_file()})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "a"
    assert data["method"] == "alphabet model only"


def test_combined_rejects_requests_without_any_input(client):
    response = client.post("/api/predict/combined")
    assert response.status_code == 400
    assert "at least one input" in response.json()["detail"]


def test_combined_reports_bad_landmarks_per_source_without_failing_others(client):
    response = client.post(
        "/api/predict/combined",
        files={"image": _png_file()},
        data={"landmarks": json.dumps([0.1] * 10)},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "a"
    fingerspelling = next(source for source in data["sources"] if source["model"] == "fingerspelling")
    assert fingerspelling["ran"] is True
    assert fingerspelling["ok"] is False
    assert str(FEATURE_DIM) in fingerspelling["detail"]


def test_combined_all_inputs_invalid_returns_400(client):
    response = client.post("/api/predict/combined", data={"landmarks": "not-json"})
    assert response.status_code == 400
    assert "Combined recognition failed" in response.json()["detail"]


def test_combined_skips_unavailable_word_model(client, monkeypatch):
    monkeypatch.setattr(main, "word_predictor", StubWordPredictor(available=False))
    response = client.post(
        "/api/predict/combined", files={"video": _video_file()}, data={"landmarks": _landmarks()}
    )
    assert response.status_code == 200
    data = response.json()
    word = next(source for source in data["sources"] if source["model"] == "word")
    assert word["ran"] is True and word["ok"] is False
    assert "not installed" in word["detail"]
    # The letter models still produced a consensus.
    assert data["label"] == "a"


def test_combined_word_only_with_unavailable_model_fails(client, monkeypatch):
    monkeypatch.setattr(main, "word_predictor", StubWordPredictor(available=False))
    response = client.post("/api/predict/combined", files={"video": _video_file()})
    assert response.status_code == 503


def test_combined_word_only_uses_word_model(client):
    response = client.post("/api/predict/combined", files={"video": _video_file()})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "hello"
    assert data["word"]["confidence"] == pytest.approx(0.8)
