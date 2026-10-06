"""API wiring tests for the continuous (live-stream) recognition module.

Uses a stub fingerspelling predictor (overridden via FastAPI's own
``app.dependency_overrides``) so these tests run without TensorFlow, exactly
like ``tests/test_api.py``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from app.main import app  # noqa: E402
from app.routes import continuous  # noqa: E402
from app.services.keras_classifier import Prediction  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

FEATURE_DIM = 126


class SequencedStubPredictor:
    """Returns a scripted sequence of top-1 labels, one call at a time."""

    available = True

    def __init__(self, labels):
        self._labels = iter(labels)

    def predict(self, vector, top_k=3):
        label = next(self._labels)
        return [Prediction(label=label, confidence=0.95), Prediction(label="z", confidence=0.02)]


@pytest.fixture()
def client():
    return TestClient(app)


def _hand_vector(value: float = 0.5) -> list[float]:
    vector = [0.0] * FEATURE_DIM
    vector[0] = value
    return vector


def _override(predictor):
    app.dependency_overrides[continuous.get_predictor] = lambda: predictor


def teardown_function(_function):
    app.dependency_overrides.pop(continuous.get_predictor, None)


def test_session_lifecycle_creates_and_reports_empty_state(client):
    session_id = client.post("/api/continuous/session").json()["session_id"]
    state = client.get(f"/api/continuous/session/{session_id}").json()
    assert state == {"text": "", "tokens": []}


def test_unknown_session_returns_404(client):
    response = client.get("/api/continuous/session/does-not-exist")
    assert response.status_code == 404


def test_frame_with_no_hand_is_reported_and_resets_state(client):
    _override(SequencedStubPredictor(["a"]))
    session_id = client.post("/api/continuous/session").json()["session_id"]
    response = client.post(f"/api/continuous/session/{session_id}/frame", json={"landmarks": [0.0] * FEATURE_DIM})
    assert response.status_code == 200
    data = response.json()
    assert data["accepted"] is False
    assert data["reason"] == "No hand detected"


def test_stable_hand_sign_is_accepted_once_buffer_and_hold_are_satisfied(client):
    # BUFFER_FRAMES=4 raw frames must land before a prediction is even made,
    # then LabelStabilizer needs its own hold streak (default 3) of the same
    # label before accepting it -- so send enough identical frames to clear
    # both and confirm exactly one acceptance, not one per frame.
    _override(SequencedStubPredictor(["a"] * 10))
    session_id = client.post("/api/continuous/session").json()["session_id"]
    accepted = []
    for _ in range(10):
        data = client.post(f"/api/continuous/session/{session_id}/frame", json={"landmarks": _hand_vector()}).json()
        accepted.append(data["accepted"])
    assert accepted.count(True) == 1
    state = client.get(f"/api/continuous/session/{session_id}").json()
    assert state["text"] == "a"
    assert state["tokens"] == ["a"]


def test_space_backspace_and_clear_use_the_sentence_processor(client):
    session_id = client.post("/api/continuous/session").json()["session_id"]
    client.post(f"/api/continuous/session/{session_id}/punctuation", json={"mark": "."})
    state = client.get(f"/api/continuous/session/{session_id}").json()
    assert state["tokens"] == ["period"]

    backspaced = client.post(f"/api/continuous/session/{session_id}/backspace").json()
    assert backspaced["tokens"] == []

    client.post(f"/api/continuous/session/{session_id}/space")
    cleared = client.post(f"/api/continuous/session/{session_id}/clear").json()
    assert cleared == {"text": "", "tokens": []}


def test_frame_rejects_unavailable_model(client):
    class Unavailable:
        available = False

    _override(Unavailable())
    session_id = client.post("/api/continuous/session").json()["session_id"]
    response = client.post(f"/api/continuous/session/{session_id}/frame", json={"landmarks": _hand_vector()})
    assert response.status_code == 503
