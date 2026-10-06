"""API wiring tests for the translation and emergency-phrase modules."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from app.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(app)


def test_languages_endpoint_lists_supported_languages(client):
    data = client.get("/api/languages").json()
    codes = {entry["code"] for entry in data["languages"]}
    assert {"ta", "hi"} <= codes


def test_translate_endpoint_translates_a_known_phrase(client):
    response = client.post("/api/translate", json={"text": "thank you", "language": "ta"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["translated"] == "நன்றி"


def test_translate_endpoint_reports_unresolved_words(client):
    response = client.post("/api/translate", json={"text": "hello spaceship", "language": "ta"})
    data = response.json()
    assert data["success"] is False
    assert data["unresolved"] == ["spaceship"]


def test_emergency_phrases_are_listed_and_translated(client):
    response = client.get("/api/emergency/phrases", params={"language": "hi"})
    assert response.status_code == 200
    data = response.json()
    assert "not real" in data["note"].lower() or "prototype" in data["note"].lower()
    help_phrase = next(p for p in data["phrases"] if p["id"] == "help")
    assert help_phrase["translated"] == "मुझे मदद चाहिए"


def test_emergency_alert_is_acknowledged_but_not_a_real_dispatch(client):
    response = client.post("/api/emergency/alert", json={"phrase_id": "ambulance", "language": "ta"})
    assert response.status_code == 200
    data = response.json()
    assert data["acknowledged"] is True
    assert data["translated"] == "ஆம்புலன்ஸை அழைக்கவும்"
    assert "no real emergency service" in data["note"].lower()

    log = client.get("/api/emergency/log").json()
    assert log["alerts"][-1]["phrase_id"] == "ambulance"


def test_emergency_alert_rejects_unknown_phrase_id(client):
    response = client.post("/api/emergency/alert", json={"phrase_id": "nope"})
    assert response.status_code == 404
