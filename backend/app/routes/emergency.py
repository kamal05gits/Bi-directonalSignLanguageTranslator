"""Emergency-phrase API.

Lists pre-translated, high-value phrases for fast on-screen/spoken use in a
crisis, keeps an in-memory log of raised alerts, and - when Twilio is
configured via environment variables - sends a real SMS and places a real
voice call to one fixed emergency contact number through
:class:`app.services.notifications.TwilioNotifier`. When Twilio is not
configured, ``POST /api/emergency/alert`` behaves exactly like the original
prototype: it records the alert in memory and says plainly that no real
emergency service was contacted.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..language.translator import DictionaryTranslator
from ..services.emergency import PHRASES, EmergencyLog, find_phrase
from ..services.notifications import TwilioNotifier

NOT_REAL_DISPATCH_NOTE = (
    "Prototype only: no real emergency service, SMS, or phone call was placed. "
    "This alert was recorded in memory for demo purposes only."
)
TWILIO_NOT_CONFIGURED_NOTE = (
    NOT_REAL_DISPATCH_NOTE
    + " Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, and "
    "TWILIO_ALERT_TO_NUMBER to enable real SMS/call dispatch."
)


def _translate_phrase(translator: DictionaryTranslator, text: str, language: str) -> str:
    if language == "en":
        return text
    result = translator.translate(text, language)
    return result.text if result.success else text


class AlertRequest(BaseModel):
    phrase_id: str
    language: str = "en"


class AlertResponse(BaseModel):
    acknowledged: bool
    phrase: str
    translated: str
    note: str
    dispatched: bool
    sms_sid: str | None = None
    call_sid: str | None = None
    dispatch_errors: list[str] = []


def build_router(translator: DictionaryTranslator, notifier: TwilioNotifier | None = None) -> APIRouter:
    router = APIRouter(prefix="/api/emergency", tags=["emergency"])
    log = EmergencyLog()
    notifier = notifier or TwilioNotifier()

    @router.get("/phrases")
    def list_phrases(language: str = "en") -> dict:
        return {
            "language": language,
            "note": NOT_REAL_DISPATCH_NOTE if not notifier.configured else (
                "Twilio dispatch is configured: raising an alert sends a real SMS/call "
                "to the configured emergency contact."
            ),
            "phrases": [
                {
                    "id": phrase.id,
                    "text": phrase.text,
                    "translated": _translate_phrase(translator, phrase.text, language),
                }
                for phrase in PHRASES
            ],
        }

    @router.get("/status")
    def dispatch_status() -> dict:
        """Report whether real Twilio dispatch is configured, without leaking secrets."""
        return {
            "twilio_configured": notifier.configured,
            "sms_enabled": notifier.settings.enable_sms,
            "call_enabled": notifier.settings.enable_call,
        }

    @router.post("/alert", response_model=AlertResponse)
    def raise_alert(payload: AlertRequest) -> AlertResponse:
        phrase = find_phrase(payload.phrase_id)
        if phrase is None:
            raise HTTPException(404, f"Unknown emergency phrase id '{payload.phrase_id}'.")

        translated = _translate_phrase(translator, phrase.text, payload.language)
        dispatch = notifier.send_alert(f"SignBridge emergency alert: {phrase.text} ({translated})")

        log.record(
            payload.phrase_id,
            payload.language,
            dispatched=dispatch.success,
            sms_sid=dispatch.sms_sid,
            call_sid=dispatch.call_sid,
            dispatch_errors=dispatch.errors,
        )

        if not dispatch.attempted:
            note = TWILIO_NOT_CONFIGURED_NOTE
        elif dispatch.success:
            parts = []
            if dispatch.sms_sid:
                parts.append(f"SMS {dispatch.sms_sid}")
            if dispatch.call_sid:
                parts.append(f"call {dispatch.call_sid}")
            note = "Real alert dispatched via Twilio (" + ", ".join(parts) + ") to the configured emergency contact."
        else:
            note = "Twilio dispatch was attempted but failed (" + "; ".join(dispatch.errors) + "). No real emergency service was reached."

        return AlertResponse(
            acknowledged=True,
            phrase=phrase.text,
            translated=translated,
            note=note,
            dispatched=dispatch.success,
            sms_sid=dispatch.sms_sid,
            call_sid=dispatch.call_sid,
            dispatch_errors=dispatch.errors,
        )

    @router.get("/log")
    def recent_alerts() -> dict:
        return {
            "alerts": [
                {
                    "phrase_id": alert.phrase_id,
                    "language": alert.language,
                    "timestamp": alert.timestamp,
                    "dispatched": alert.dispatched,
                    "sms_sid": alert.sms_sid,
                    "call_sid": alert.call_sid,
                    "dispatch_errors": alert.dispatch_errors,
                }
                for alert in log.recent()
            ]
        }

    return router
