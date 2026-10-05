"""Optional real emergency dispatch via Twilio (SMS + voice call).

This is the "real provider" extension point the emergency module always
mentioned: when the required environment variables are configured, raising
an emergency alert also sends a real SMS and places a real voice call
(using Twilio's text-to-speech `<Say>`) to one fixed, operator-configured
contact number. When they are not configured - e.g. local development, or a
deployment that has not set up a Twilio account - :class:`TwilioNotifier`
reports itself as unconfigured and the caller keeps its previous
in-memory-only prototype behaviour. Nothing here ever pretends to have sent
an alert that was not actually sent.

Required environment variables for real dispatch:

- ``TWILIO_ACCOUNT_SID`` - Twilio Account SID.
- ``TWILIO_AUTH_TOKEN`` - Twilio Auth Token.
- ``TWILIO_FROM_NUMBER`` - A Twilio phone number (E.164, e.g. ``+15551234567``).
- ``TWILIO_ALERT_TO_NUMBER`` - The emergency contact's phone number (E.164)
  that receives the SMS/call. Fixed for the whole deployment by design.

Optional:

- ``TWILIO_ENABLE_SMS`` (default ``true``)
- ``TWILIO_ENABLE_CALL`` (default ``true``)
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any
from xml.sax.saxutils import escape


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class TwilioSettings:
    account_sid: str | None
    auth_token: str | None
    from_number: str | None
    to_number: str | None
    enable_sms: bool = True
    enable_call: bool = True

    @classmethod
    def from_env(cls) -> "TwilioSettings":
        return cls(
            account_sid=os.getenv("TWILIO_ACCOUNT_SID") or None,
            auth_token=os.getenv("TWILIO_AUTH_TOKEN") or None,
            from_number=os.getenv("TWILIO_FROM_NUMBER") or None,
            to_number=os.getenv("TWILIO_ALERT_TO_NUMBER") or None,
            enable_sms=_env_flag("TWILIO_ENABLE_SMS", True),
            enable_call=_env_flag("TWILIO_ENABLE_CALL", True),
        )

    @property
    def configured(self) -> bool:
        """True once all four required values are present."""
        return bool(self.account_sid and self.auth_token and self.from_number and self.to_number)


@dataclass
class DispatchResult:
    attempted: bool
    sms_sid: str | None = None
    call_sid: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.attempted and not self.errors and bool(self.sms_sid or self.call_sid)


class TwilioNotifier:
    """Thread-safe, lazily-connected Twilio SMS + voice-call sender."""

    def __init__(self, settings: TwilioSettings | None = None) -> None:
        self.settings = settings or TwilioSettings.from_env()
        self._client: Any = None
        self._lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return self.settings.configured

    def _load_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is None:
                try:
                    from twilio.rest import Client
                except ImportError as exc:
                    raise RuntimeError(
                        "The twilio package is not installed. Run: pip install -r backend/requirements.txt"
                    ) from exc
                self._client = Client(self.settings.account_sid, self.settings.auth_token)
        return self._client

    def send_alert(self, message: str) -> DispatchResult:
        """Send ``message`` by SMS and/or voice call to the configured contact.

        Returns a :class:`DispatchResult` with ``attempted=False`` when Twilio
        is not configured at all, so callers can tell "we didn't try" apart
        from "we tried and it failed" (``attempted=True`` with ``errors``).
        """
        if not self.settings.configured:
            return DispatchResult(attempted=False)

        result = DispatchResult(attempted=True)
        try:
            client = self._load_client()
        except RuntimeError as exc:
            result.errors.append(str(exc))
            return result

        if self.settings.enable_sms:
            try:
                sms = client.messages.create(
                    body=message,
                    from_=self.settings.from_number,
                    to=self.settings.to_number,
                )
                result.sms_sid = sms.sid
            except Exception as exc:  # Twilio raises several exception types for API/auth failures.
                result.errors.append(f"SMS failed: {exc}")

        if self.settings.enable_call:
            try:
                twiml = f"<Response><Say>{escape(message)}</Say></Response>"
                call = client.calls.create(
                    twiml=twiml,
                    from_=self.settings.from_number,
                    to=self.settings.to_number,
                )
                result.call_sid = call.sid
            except Exception as exc:
                result.errors.append(f"Call failed: {exc}")

        return result
