"""API wiring tests for the three model endpoints.

Skipped automatically in environments without FastAPI (e.g. the minimal CI
job), and uses stub predictors so no model files or heavy frameworks load.
"""

import io

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
import app.main as main  # noqa: E402
from app.services.model_base import Prediction  # noqa: E402


class StubPredictor:
    labels = ["a", "b"]
    loaded = True
    available = True

    def predict(self, value, top_k=3):
        return [Prediction(label="a", confidence=0.9), Prediction(label="b", confidence=0.1)]

    def predict_video(self, payload, suffix, top_k=5):
        return [Prediction(label="hello", confidence=0.8), Prediction(label="yes", confidence=0.2)]

    def availability(self):
        return True, "ready"


@pytest.fixture(autouse=True)
def stub_predictors(monkeypatch):
    monkeypatch.setattr(main, "alphabet_predictor", StubPredictor())
    monkeypatch.setattr(main, "fingerspelling_predictor", StubPredictor())
    monkeypatch.setattr(main, "word_predictor", StubPredictor())
    yield


@pytest.fixture()
def client():
    return TestClient(app)


def _png_bytes():
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 255, 255)).save(buffer, "PNG")
    return buffer.getvalue()


def test_health_reports_all_three_models(client):
    data = client.get("/api/health").json()
    assert data["status"] == "ok"
    assert set(data["models"]) == {"alphabet", "fingerspelling", "word"}
    assert all("detail" in entry for entry in data["models"].values())


def test_alphabet_predict_accepts_image(client):
    response = client.post("/api/predict/alphabet", files={"file": ("hand.png", _png_bytes(), "image/png")})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "a"
    assert data["accepted"] is True
    assert len(data["top_predictions"]) == 2


def test_alphabet_predict_rejects_non_image(client):
    response = client.post("/api/predict/alphabet", files={"file": ("x.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_fingerspelling_predict_accepts_landmarks(client):
    landmarks = [0.001 * i for i in range(126)]
    response = client.post("/api/predict/fingerspelling", json={"landmarks": landmarks})
    assert response.status_code == 200
    assert response.json()["accepted"] is True


def test_fingerspelling_predict_rejects_short_vector(client):
    response = client.post("/api/predict/fingerspelling", json={"landmarks": [0.0, 1.0]})
    assert response.status_code == 422


def test_word_predict_accepts_video(client):
    response = client.post("/api/predict/word", files={"file": ("sign.webm", b"\x1a\x45\xdf\xa3fakevideo", "video/webm")})
    assert response.status_code == 200
    assert response.json()["label"] == "hello"


def test_word_predict_rejects_non_video(client):
    response = client.post("/api/predict/word", files={"file": ("hand.png", _png_bytes(), "image/png")})
    assert response.status_code == 415


def test_info_lists_models(client):
    data = client.get("/api/info").json()
    assert set(data["models"]) == {"alphabet", "fingerspelling", "word"}
