"""Unit tests for the optional Twilio emergency-dispatch integration.

These avoid the real ``twilio`` package and network calls entirely: they
either leave Twilio unconfigured (asserting the graceful prototype
fallback) or monkeypatch ``TwilioNotifier._load_client`` with an in-memory
fake client, matching the project's "never fabricate, never require the
real dependency in tests" convention.
"""

from __future__ import annotations

from app.services.notifications import DispatchResult, TwilioNotifier, TwilioSettings


def _settings(**overrides):
    base = dict(
        account_sid="AC123",
        auth_token="token",
        from_number="+15550000000",
        to_number="+15551111111",
        enable_sms=True,
        enable_call=True,
    )
    base.update(overrides)
    return TwilioSettings(**base)


def test_settings_from_env_reports_unconfigured_when_missing(monkeypatch):
    for key in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER", "TWILIO_ALERT_TO_NUMBER"):
        monkeypatch.delenv(key, raising=False)
    settings = TwilioSettings.from_env()
    assert not settings.configured


def test_settings_from_env_reads_all_required_values(monkeypatch):
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret")
    monkeypatch.setenv("TWILIO_FROM_NUMBER", "+15550000000")
    monkeypatch.setenv("TWILIO_ALERT_TO_NUMBER", "+15551111111")
    settings = TwilioSettings.from_env()
    assert settings.configured
    assert settings.enable_sms and settings.enable_call


def test_settings_from_env_honors_disable_flags(monkeypatch):
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret")
    monkeypatch.setenv("TWILIO_FROM_NUMBER", "+15550000000")
    monkeypatch.setenv("TWILIO_ALERT_TO_NUMBER", "+15551111111")
    monkeypatch.setenv("TWILIO_ENABLE_CALL", "false")
    settings = TwilioSettings.from_env()
    assert settings.enable_sms is True
    assert settings.enable_call is False


def test_send_alert_does_not_attempt_when_unconfigured():
    notifier = TwilioNotifier(settings=_settings(account_sid=None))
    result = notifier.send_alert("help")
    assert result.attempted is False
    assert result.success is False


class _FakeMessages:
    def __init__(self, sid="SM123"):
        self.sid = sid
        self.calls_made = []

    def create(self, body, from_, to):
        self.calls_made.append((body, from_, to))

        class _Message:
            sid = self.sid

        return _Message()


class _FakeCalls:
    def __init__(self, sid="CA123"):
        self.sid = sid
        self.calls_made = []

    def create(self, twiml, from_, to):
        self.calls_made.append((twiml, from_, to))

        class _Call:
            sid = self.sid

        return _Call()


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()
        self.calls = _FakeCalls()


def test_send_alert_sends_sms_and_call_when_configured(monkeypatch):
    notifier = TwilioNotifier(settings=_settings())
    fake_client = _FakeClient()
    monkeypatch.setattr(notifier, "_load_client", lambda: fake_client)

    result = notifier.send_alert("I need help")

    assert result.attempted is True
    assert result.success is True
    assert result.sms_sid == "SM123"
    assert result.call_sid == "CA123"
    assert fake_client.messages.calls_made == [("I need help", "+15550000000", "+15551111111")]
    assert fake_client.calls.calls_made[0][1:] == ("+15550000000", "+15551111111")
    assert "<Say>I need help</Say>" in fake_client.calls.calls_made[0][0]


def test_send_alert_respects_disabled_channels(monkeypatch):
    notifier = TwilioNotifier(settings=_settings(enable_call=False))
    fake_client = _FakeClient()
    monkeypatch.setattr(notifier, "_load_client", lambda: fake_client)

    result = notifier.send_alert("help")

    assert result.sms_sid == "SM123"
    assert result.call_sid is None
    assert not fake_client.calls.calls_made


def test_send_alert_collects_errors_without_raising(monkeypatch):
    notifier = TwilioNotifier(settings=_settings())

    class _FailingClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("boom")

        class calls:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("also boom")

    monkeypatch.setattr(notifier, "_load_client", lambda: _FailingClient())
    result = notifier.send_alert("help")

    assert result.attempted is True
    assert result.success is False
    assert len(result.errors) == 2


def test_dispatch_result_success_requires_a_sid():
    assert DispatchResult(attempted=True).success is False
    assert DispatchResult(attempted=True, sms_sid="SM1").success is True
