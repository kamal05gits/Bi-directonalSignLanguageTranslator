"""Twilio SMS + voice-call notifications for emergency alerts.

Configuration (environment variables):

==============================  ==================================================
``TWILIO_ACCOUNT_SID``           Twilio Account SID (required)
``TWILIO_AUTH_TOKEN``            Twilio Auth Token (required)
``EMERGENCY_TO_NUMBER``          The phone number alerts go to, E.164 (required)
``TWILIO_FROM_NUMBER``           Twilio number placing the SMS/call, E.164
``TWILIO_MESSAGING_SERVICE_SID`` Optional Messaging Service (alternative to
                                 ``TWILIO_FROM_NUMBER`` for SMS; the voice call
                                 still needs ``TWILIO_FROM_NUMBER``)
``EMERGENCY_SMS_ENABLED``        Send an SMS (default ``true``)
``EMERGENCY_VOICE_CALL_ENABLED`` Place a voice call (default ``true``)
==============================  ==================================================

The integration degrades the same way the word model does: when the
``twilio`` package is not installed or the credentials above are missing,
emergency alerts keep working as the on-screen/spoken prototype, and every
API response says so plainly instead of pretending a message was sent.

Numbers are masked (``+91••••••210``) in API responses so the configured
phone number is never echoed back to the browser.
"""

from __future__ import annotations

import importlib.util
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from xml.sax.saxutils import escape

from .keras_classifier import Prediction  # noqa: F401  (keeps the services layer import graph simple)

# <Say> language/voice pairs for translated phrases. English always works;
# other languages fall back to the English phrase so the call never fails on
# an unsupported voice.
SAY_LANGUAGES: dict[str, tuple[str, str]] = {
    "en": ("en-US", ""),
    "hi": ("hi-IN", "Polly.Aditi"),
}

E164_HINT = "Phone numbers must be in E.164 format, e.g. +919876543210."


def _is_e164(value: str) -> bool:
    return value.startswith("+") and value[1:].isdigit() and 8 <= len(value) <= 16


def mask_number(number: str) -> str:
    """Mask a phone number for display: ``+919876543210`` -> ``+91••••••210``."""
    if len(number) <= 5:
        return "•" * len(number)
    return f"{number[:3]}{'•' * (len(number) - 6)}{number[-3:]}"


@dataclass(frozen=True)
class TwilioConfig:
    account_sid: str
    auth_token: str
    from_number: str
    messaging_service_sid: str
    to_number: str
    sms_enabled: bool
    call_enabled: bool

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "TwilioConfig | None":
        """Build a config from the environment, or ``None`` when incomplete."""
        env = os.environ if env is None else env
        sid = env.get("TWILIO_ACCOUNT_SID", "").strip()
        token = env.get("TWILIO_AUTH_TOKEN", "").strip()
        to_number = env.get("EMERGENCY_TO_NUMBER", "").strip()
        from_number = env.get("TWILIO_FROM_NUMBER", "").strip()
        messaging = env.get("TWILIO_MESSAGING_SERVICE_SID", "").strip()
        if not (sid and token and to_number and (from_number or messaging)):
            return None
        return cls(
            account_sid=sid,
            auth_token=token,
            from_number=from_number,
            messaging_service_sid=messaging,
            to_number=to_number,
            sms_enabled=env.get("EMERGENCY_SMS_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"},
            call_enabled=env.get("EMERGENCY_VOICE_CALL_ENABLED", "true").strip().lower()
            not in {"0", "false", "no", "off"},
        )

    def validate(self) -> str | None:
        """Return a human-readable problem with the numbers, or None."""
        if not _is_e164(self.to_number):
            return f"EMERGENCY_TO_NUMBER is invalid. {E164_HINT}"
        if self.sms_enabled and not (self.from_number or self.messaging_service_sid):
            return "SMS needs TWILIO_FROM_NUMBER or TWILIO_MESSAGING_SERVICE_SID."
        if self.call_enabled and not self.from_number:
            return "Voice calls need TWILIO_FROM_NUMBER (a Twilio number that supports voice)."
        return None


@dataclass(frozen=True)
class ChannelResult:
    channel: str  # "sms" | "call"
    sent: bool
    detail: str
    sid: str | None = None


@dataclass(frozen=True)
class NotificationOutcome:
    provider: str  # "twilio" | "none"
    configured: bool
    to_masked: str | None
    channels: list[ChannelResult] = field(default_factory=list)
    sdk_missing: bool = False

    @property
    def delivery(self) -> str:
        """Compact delivery tag for the alert log, e.g. ``"sms+call"``."""
        sent = [result.channel for result in self.channels if result.sent]
        return "+".join(sent) if sent else "none"

    @property
    def summary(self) -> str:
        """One sentence describing what happened, for API notes."""
        if self.provider == "none":
            return "No notification provider is configured."
        if self.sdk_missing:
            return "Twilio credentials are set, but the twilio package is not installed (pip install twilio), so nothing was sent."
        if not self.channels:
            return "All notification channels are disabled."
        parts = [
            f"{result.channel.upper()} {'sent' if result.sent else 'failed'}"
            + ("" if result.sent else f" ({result.detail})")
            for result in self.channels
        ]
        return "Twilio: " + ", ".join(parts) + f" → {self.to_masked}"


def _default_client_factory(config: TwilioConfig):
    """Import twilio lazily so the service works without the package."""
    if importlib.util.find_spec("twilio") is None:
        raise ImportError("The twilio package is not installed. Add it with: pip install twilio")
    from twilio.rest import Client

    return Client(config.account_sid, config.auth_token)


def build_sms_body(phrase_en: str, phrase_translated: str, context: str | None, when: str) -> str:
    lines = ["SignBridge EMERGENCY ALERT", f'"{phrase_en}"']
    if phrase_translated and phrase_translated != phrase_en:
        lines.append(f"({phrase_translated})")
    if context:
        lines.append(f'Sender\'s message: "{context}"')
    lines.append(f"{when} — SignBridge ISL translator")
    return "\n".join(lines)


def build_call_twiml(phrase_en: str, phrase_translated: str, language: str) -> str:
    """TwiML that speaks the alert when the call is answered."""
    say_language, voice = SAY_LANGUAGES.get(language, SAY_LANGUAGES["en"])
    say_attrs = f' language="{say_language}"' + (f' voice="{voice}"' if voice else "")
    parts = [
        "<Response>",
        '<Pause length="1"/>',
        f"<Say{say_attrs}>This is an emergency alert from SignBridge.</Say>",
        f"<Say{say_attrs}>{escape(phrase_en)}.</Say>",
    ]
    if phrase_translated and phrase_translated != phrase_en:
        translated_language, translated_voice = SAY_LANGUAGES.get(language, ("", ""))
        if translated_language and translated_language != "en-US":
            attrs = f' language="{translated_language}"'
            if translated_voice:
                attrs += f' voice="{translated_voice}"'
            parts.append(f"<Say{attrs}>{escape(phrase_translated)}.</Say>")
    parts.append(f"<Say{say_attrs}>The sender may be deaf or hard of hearing. Please check on them.</Say>")
    parts.append("</Response>")
    return "".join(parts)


class TwilioNotifier:
    """Send emergency alerts over Twilio SMS and voice, or report honestly."""

    def __init__(self, config: TwilioConfig | None, client_factory: Callable[[TwilioConfig], object] | None = None):
        self._config = config
        self._client_factory = client_factory or _default_client_factory
        self._client: object | None = None

    @property
    def configured(self) -> bool:
        return self._config is not None

    @property
    def config(self) -> TwilioConfig | None:
        return self._config

    def sdk_available(self) -> bool:
        return importlib.util.find_spec("twilio") is not None

    def status(self) -> dict[str, object]:
        """Provider status for ``GET /api/emergency/status`` and the UI."""
        if not self._config:
            return {
                "provider": "none",
                "configured": False,
                "sms_enabled": False,
                "call_enabled": False,
                "to_masked": None,
                "sdk_available": self.sdk_available(),
                "note": "No SMS or call can be sent: Twilio is not configured. "
                "Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER (or "
                "TWILIO_MESSAGING_SERVICE_SID), and EMERGENCY_TO_NUMBER.",
            }
        problem = self._config.validate()
        channels = []
        if self._config.sms_enabled:
            channels.append("sms")
        if self._config.call_enabled:
            channels.append("call")
        note = (
            f"Alerts are sent by {' and '.join(channels) if channels else 'no channels'} "
            f"to {mask_number(self._config.to_number)} via Twilio."
        )
        if not self.sdk_available():
            note = "Twilio is configured, but the twilio package is not installed — nothing can be sent yet."
        elif problem:
            note = f"Twilio is configured incorrectly: {problem}"
        return {
            "provider": "twilio",
            "configured": True,
            "sms_enabled": self._config.sms_enabled,
            "call_enabled": self._config.call_enabled,
            "to_masked": mask_number(self._config.to_number),
            "sdk_available": self.sdk_available(),
            "note": note,
        }

    def notify(
        self,
        *,
        phrase_en: str,
        phrase_translated: str = "",
        language: str = "en",
        context: str | None = None,
    ) -> NotificationOutcome:
        """Try to deliver the alert; never raises."""
        if not self._config:
            return NotificationOutcome(provider="none", configured=False, to_masked=None)
        config = self._config
        masked = mask_number(config.to_number)
        when = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

        problem = config.validate()
        if problem:
            failed = [
                ChannelResult("sms" if channel == "sms" else "call", False, problem)
                for channel in (["sms"] if config.sms_enabled else [])
                + (["call"] if config.call_enabled else [])
            ]
            return NotificationOutcome(
                provider="twilio",
                configured=True,
                to_masked=masked,
                channels=failed,
            )

        try:
            if self._client is None:
                self._client = self._client_factory(config)
        except ImportError as exc:
            return NotificationOutcome(
                provider="twilio",
                configured=True,
                to_masked=masked,
                sdk_missing=True,
                channels=[ChannelResult("sms", False, str(exc))],
            )

        channels: list[ChannelResult] = []
        if config.sms_enabled:
            channels.append(self._send_sms(config, phrase_en, phrase_translated, context, when, masked))
        if config.call_enabled:
            channels.append(self._place_call(config, phrase_en, phrase_translated, language, masked))
        return NotificationOutcome(
            provider="twilio", configured=True, to_masked=masked, channels=channels
        )

    # ------------------------------------------------------------------ channels

    def _send_sms(
        self,
        config: TwilioConfig,
        phrase_en: str,
        phrase_translated: str,
        context: str | None,
        when: str,
        masked: str,
    ) -> ChannelResult:
        body = build_sms_body(phrase_en, phrase_translated, context, when)
        kwargs: dict[str, str] = {"to": config.to_number, "body": body}
        if config.messaging_service_sid:
            kwargs["messaging_service_sid"] = config.messaging_service_sid
        else:
            kwargs["from_"] = config.from_number
        try:
            message = self._client.messages.create(**kwargs)
            return ChannelResult("sms", True, f"SMS sent to {masked}", getattr(message, "sid", None))
        except Exception as exc:  # Twilio raises many exception types; delivery must never crash the alert.
            return ChannelResult("sms", False, f"SMS failed: {exc}")

    def _place_call(
        self,
        config: TwilioConfig,
        phrase_en: str,
        phrase_translated: str,
        language: str,
        masked: str,
    ) -> ChannelResult:
        twiml = build_call_twiml(phrase_en, phrase_translated, language)
        try:
            call = self._client.calls.create(
                to=config.to_number, from_=config.from_number, twiml=twiml
            )
            return ChannelResult("call", True, f"Voice call placed to {masked}", getattr(call, "sid", None))
        except Exception as exc:
            return ChannelResult("call", False, f"Voice call failed: {exc}")
