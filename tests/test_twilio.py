"""Tests for the Twilio emergency notifier and its emergency-API wiring.

All Twilio traffic goes through a fake client, so these tests need no
credentials and make no network calls.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.routes import emergency as emergency_routes  # noqa: E402
from app.services.twilio_notifier import (  # noqa: E402
    TwilioConfig,
    TwilioNotifier,
    build_call_twiml,
    build_sms_body,
    mask_number,
)


class FakeTwilioEndpoint:
    def __init__(self, sid_prefix: str, fail: bool = False) -> None:
        self.sid_prefix = sid_prefix
        self.fail = fail
        self.created: list[dict] = []

    def create(self, **kwargs):
        if self.fail:
            raise Exception("boom")
        self.created.append(kwargs)
        return SimpleNamespace(sid=f"{self.sid_prefix}123")


class FakeTwilioClient:
    def __init__(self, fail_sms: bool = False, fail_call: bool = False) -> None:
        self.messages = FakeTwilioEndpoint("SM", fail_sms)
        self.calls = FakeTwilioEndpoint("CA", fail_call)


def _config(**overrides) -> TwilioConfig:
    values = dict(
        account_sid="ACtest",
        auth_token="secret",
        from_number="+15550001111",
        messaging_service_sid="",
        to_number="+919876543210",
        sms_enabled=True,
        call_enabled=True,
    )
    values.update(overrides)
    return TwilioConfig(**values)


# ------------------------------------------------------------------- config


def test_from_env_returns_none_when_incomplete(monkeypatch):
    keys = (
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_FROM_NUMBER",
        "TWILIO_MESSAGING_SERVICE_SID",
        "EMERGENCY_TO_NUMBER",
    )
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    assert TwilioConfig.from_env() is None
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACtest")
    assert TwilioConfig.from_env() is None  # still missing the token and numbers


def test_from_env_builds_a_full_config(monkeypatch):
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACtest")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret")
    monkeypatch.setenv("TWILIO_FROM_NUMBER", "+15550001111")
    monkeypatch.setenv("EMERGENCY_TO_NUMBER", "+919876543210")
    monkeypatch.setenv("EMERGENCY_VOICE_CALL_ENABLED", "false")
    config = TwilioConfig.from_env()
    assert config is not None
    assert config.to_number == "+919876543210"
    assert config.sms_enabled is True
    assert config.call_enabled is False


def test_validate_rejects_non_e164_destination():
    assert "EMERGENCY_TO_NUMBER" in _config(to_number="98765").validate()


def test_mask_number_hides_the_middle():
    assert mask_number("+919876543210") == "+91•••••••210"
    assert mask_number("+1234") == "•••••"


# --------------------------------------------------------------- notification


def test_notify_sends_sms_and_places_call():
    fake = FakeTwilioClient()
    notifier = TwilioNotifier(_config(), client_factory=lambda config: fake)
    outcome = notifier.notify(phrase_en="I need help", phrase_translated="உதவி வேண்டும்", language="ta")
    assert outcome.provider == "twilio"
    assert outcome.delivery == "sms+call"
    assert outcome.to_masked == "+91•••••••210"
    assert len(fake.messages.created) == 1
    sms = fake.messages.created[0]
    assert sms["to"] == "+919876543210"
    assert sms["from_"] == "+15550001111"
    assert "I need help" in sms["body"]
    assert "உதவி வேண்டும்" in sms["body"]
    call = fake.calls.created[0]
    assert call["twiml"].startswith("<Response>")
    assert "I need help" in call["twiml"]


def test_notify_uses_messaging_service_for_sms_when_set():
    fake = FakeTwilioClient()
    config = _config(from_number="", messaging_service_sid="MGtest", call_enabled=False)
    notifier = TwilioNotifier(config, client_factory=lambda cfg: fake)
    outcome = notifier.notify(phrase_en="there is a fire")
    assert outcome.channels[0].sent is True
    assert fake.messages.created[0]["messaging_service_sid"] == "MGtest"


def test_notify_reports_per_channel_failure_and_still_calls():
    fake = FakeTwilioClient(fail_sms=True)
    notifier = TwilioNotifier(_config(), client_factory=lambda config: fake)
    outcome = notifier.notify(phrase_en="I need help")
    sms, call = outcome.channels
    assert sms.sent is False and "boom" in sms.detail
    assert call.sent is True
    assert outcome.delivery == "call"
    assert "SMS failed" in outcome.summary


def test_notify_honors_disabled_channels():
    fake = FakeTwilioClient()
    notifier = TwilioNotifier(_config(call_enabled=False), client_factory=lambda config: fake)
    outcome = notifier.notify(phrase_en="I need help")
    assert [result.channel for result in outcome.channels] == ["sms"]
    assert fake.calls.created == []


def test_notify_reports_missing_sdk():
    def _missing(config):
        raise ImportError("No module named 'twilio'")

    notifier = TwilioNotifier(_config(), client_factory=_missing)
    outcome = notifier.notify(phrase_en="I need help")
    assert outcome.sdk_missing is True
    assert outcome.channels[0].sent is False
    assert "twilio" in outcome.summary


def test_unconfigured_notifier_reports_nothing_to_send():
    notifier = TwilioNotifier(None)
    outcome = notifier.notify(phrase_en="I need help")
    assert outcome.provider == "none"
    assert outcome.configured is False
    assert outcome.delivery == "none"


def test_sms_body_includes_context_message():
    body = build_sms_body("I need help", "உதவி வேண்டும்", "at the bus stop", "2026-10-05 10:00 UTC")
    assert "I need help" in body
    assert "at the bus stop" in body
    assert "SignBridge" in body


def test_call_twiml_speaks_translated_hindi():
    twiml = build_call_twiml("I need help", "मुझे मदद चाहिए", "hi")
    assert 'language="hi-IN"' in twiml
    assert "मुझे मदद चाहिए" in twiml


def test_call_twiml_falls_back_to_english_for_unmapped_languages():
    twiml = build_call_twiml("I need help", "உதவி வேண்டும்", "ta")
    assert "I need help" in twiml
    assert 'language="ta-IN"' not in twiml  # no mapped voice: speak English, keep the SMS in Tamil


# ---------------------------------------------------------- emergency API wiring


@pytest.fixture()
def client():
    return TestClient(app)


def _override_notifier(monkeypatch, notifier):
    monkeypatch.setitem(app.dependency_overrides, emergency_routes.get_notifier, lambda: notifier)


def test_emergency_alert_delivers_sms_and_call(client, monkeypatch):
    fake = FakeTwilioClient()
    notifier = TwilioNotifier(_config(), client_factory=lambda config: fake)
    _override_notifier(monkeypatch, notifier)

    response = client.post(
        "/api/emergency/alert",
        json={"phrase_id": "help", "language": "ta", "context_message": "I am at the bus stop"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["acknowledged"] is True
    assert "Twilio" in data["note"]
    assert "+91•••••••210" in data["note"]
    notification = data["notification"]
    assert notification["configured"] is True
    channels = {entry["channel"]: entry for entry in notification["channels"]}
    assert channels["sms"]["sent"] is True and channels["sms"]["sid"] == "SM123"
    assert channels["call"]["sent"] is True and channels["call"]["sid"] == "CA123"
    assert "bus stop" in fake.messages.created[0]["body"]

    log = client.get("/api/emergency/log").json()
    assert log["alerts"][-1]["delivery"] == "sms+call"


def test_emergency_alert_reports_partial_delivery(client, monkeypatch):
    fake = FakeTwilioClient(fail_call=True)
    notifier = TwilioNotifier(_config(), client_factory=lambda config: fake)
    _override_notifier(monkeypatch, notifier)

    response = client.post("/api/emergency/alert", json={"phrase_id": "fire", "language": "en"})
    data = response.json()
    assert response.status_code == 200
    assert "partially delivered" in data["note"]
    assert "voice call failed" in data["note"].lower().replace("voice call: failed", "voice call failed")
    channels = {entry["channel"]: entry for entry in data["notification"]["channels"]}
    assert channels["sms"]["sent"] is True
    assert channels["call"]["sent"] is False


def test_emergency_status_endpoint_reports_configuration(client, monkeypatch):
    notifier = TwilioNotifier(_config(), client_factory=lambda config: FakeTwilioClient())
    _override_notifier(monkeypatch, notifier)
    data = client.get("/api/emergency/status").json()
    assert data["provider"] == "twilio"
    assert data["configured"] is True
    assert data["sms_enabled"] is True and data["call_enabled"] is True
    assert data["to_masked"] == "+91•••••••210"


def test_emergency_phrases_include_notification_status(client):
    data = client.get("/api/emergency/phrases", params={"language": "en"}).json()
    assert data["notification"]["provider"] in {"none", "twilio"}
    assert any(phrase["id"] == "help" for phrase in data["phrases"])
