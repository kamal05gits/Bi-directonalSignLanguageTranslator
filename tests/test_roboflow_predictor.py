"""Tests for the optional Roboflow image provider without network access."""

from __future__ import annotations

import pytest

pytest.importorskip("requests")
pytest.importorskip("PIL")

from app.services.keras_classifier import Prediction  # noqa: E402
from app.services.roboflow_predictor import (  # noqa: E402
    RoboflowPredictor,
    _predictions_from_response,
)
from PIL import Image  # noqa: E402


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


def test_missing_key_disables_provider_without_network():
    predictor = RoboflowPredictor(api_key="")
    available, detail = predictor.availability()
    assert not available
    assert "ROBOFLOW_API_KEY" in detail


def test_detection_response_is_ranked_and_deduplicated():
    result = _predictions_from_response(
        {
            "predictions": [
                {"class": "hello", "confidence": 0.42},
                {"class": "A", "confidence": 0.91},
                {"class": "A", "confidence": 0.80},
            ]
        }
    )
    assert result == [Prediction(label="A", confidence=0.91), Prediction(label="hello", confidence=0.42)]


def test_predict_sends_base64_image_and_returns_predictions():
    predictor = RoboflowPredictor(api_key="server-only-key")
    session = FakeSession(FakeResponse(payload={"predictions": [{"class": "A", "confidence": 0.88}]}))
    predictor._request_session = session

    result = predictor.predict(Image.new("RGB", (16, 16), "white"))

    assert result == [Prediction(label="A", confidence=0.88)]
    args, kwargs = session.calls[0]
    assert args[0].endswith("/indian-sign-language_40/1")
    assert kwargs["data"]
    assert kwargs["params"]["confidence"] == pytest.approx(25.0)
    assert kwargs["params"]["overlap"] == pytest.approx(50.0)
    assert "server-only-key" not in repr(kwargs)


def test_http_failure_does_not_echo_api_key():
    predictor = RoboflowPredictor(api_key="server-only-key")
    predictor._request_session = FakeSession(FakeResponse(status_code=401))

    with pytest.raises(RuntimeError, match="HTTP 401") as error:
        predictor.predict(Image.new("RGB", (16, 16), "white"))
    assert "server-only-key" not in str(error.value)
