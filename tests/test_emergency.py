"""API wiring tests for the translation and emergency-phrase modules."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


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


def test_emergency_status_reports_twilio_not_configured_by_default(client):
    response = client.get("/api/emergency/status")
    assert response.status_code == 200
    assert response.json()["twilio_configured"] is False


def test_emergency_alert_dispatches_via_twilio_when_configured():
    """Wire a fake-but-configured TwilioNotifier straight into the router,
    bypassing the real `twilio` package and network calls entirely."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.language.translator import DictionaryTranslator
    from app.routes.emergency import build_router
    from app.services.notifications import TwilioNotifier, TwilioSettings

    class _FakeMessage:
        sid = "SM999"

    class _FakeCall:
        sid = "CA999"

    class _FakeClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                return _FakeMessage()

        class calls:
            @staticmethod
            def create(**kwargs):
                return _FakeCall()

    notifier = TwilioNotifier(
        settings=TwilioSettings(
            account_sid="AC1",
            auth_token="token",
            from_number="+15550000000",
            to_number="+15551111111",
        )
    )
    notifier._client = _FakeClient()

    app = FastAPI()
    app.include_router(build_router(DictionaryTranslator(), notifier=notifier))
    client = TestClient(app)

    status = client.get("/api/emergency/status").json()
    assert status["twilio_configured"] is True

    response = client.post("/api/emergency/alert", json={"phrase_id": "help"})
    data = response.json()
    assert data["dispatched"] is True
    assert data["sms_sid"] == "SM999"
    assert data["call_sid"] == "CA999"
    assert "real alert dispatched via twilio" in data["note"].lower()
