"""Emergency-phrase API with Twilio SMS/voice delivery.

One tap raises a pre-translated, high-value phrase that is shown large,
spoken aloud, recorded in the in-memory audit log, and - when Twilio is
configured via environment variables (see
:class:`app.services.twilio_notifier.TwilioConfig`) - delivered to a real
phone number as an SMS and a synthesized voice call. When Twilio is **not**
configured the alert still works on screen and every response says plainly
that nothing was sent, so the prototype never pretends to have a capability
it lacks.

The notifier is injected through a FastAPI dependency (``get_notifier``) so
``app.main`` wires the env-configured Twilio client while tests substitute a
stub without needing credentials.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..language.translator import DictionaryTranslator
from ..services.emergency import PHRASES, EmergencyLog, find_phrase
from ..services.twilio_notifier import NotificationOutcome, TwilioNotifier

NOT_REAL_DISPATCH_NOTE = (
    "Prototype only: no real emergency service, SMS, or phone call was placed. "
    "This alert was recorded in memory for demo purposes only."
)

# Kept as the fallback note so existing clients keep seeing the honest
# prototype wording whenever no provider is configured.


def _translate_phrase(translator: DictionaryTranslator, text: str, language: str) -> str:
    if language == "en":
        return text
    result = translator.translate(text, language)
    return result.text if result.success else text


class AlertRequest(BaseModel):
    phrase_id: str
    language: str = "en"
    context_message: str | None = Field(
        None, max_length=240, description="Optional message the user built, appended to the SMS."
    )


class ChannelStatus(BaseModel):
    channel: str
    sent: bool
    detail: str
    sid: str | None = None


class NotificationStatus(BaseModel):
    provider: str
    configured: bool
    to_masked: str | None = None
    channels: list[ChannelStatus] = []


class AlertResponse(BaseModel):
    acknowledged: bool
    phrase: str
    translated: str
    note: str
    notification: NotificationStatus | None = None


_default_notifier = TwilioNotifier(None)


def get_notifier() -> TwilioNotifier:
    """FastAPI dependency: the env-configured notifier, overridable in tests."""
    return _default_notifier


def _notification_status(outcome: NotificationOutcome) -> NotificationStatus:
    return NotificationStatus(
        provider=outcome.provider,
        configured=outcome.configured,
        to_masked=outcome.to_masked,
        channels=[
            ChannelStatus(channel=result.channel, sent=result.sent, detail=result.detail, sid=result.sid)
            for result in outcome.channels
        ],
    )


def _note_for(outcome: NotificationOutcome) -> str:
    if not outcome.configured:
        return NOT_REAL_DISPATCH_NOTE
    if outcome.sdk_missing:
        return (
            "Twilio credentials are set, but the twilio package is not installed "
            "(pip install twilio). No SMS or call was placed; the alert was recorded in memory only."
        )
    sent = [result for result in outcome.channels if result.sent]
    failed = [result for result in outcome.channels if not result.sent]
    masked = outcome.to_masked or "the configured number"
    if sent and not failed:
        delivered = " and ".join(
            {"sms": "SMS", "call": "voice call"}[result.channel] for result in sent
        )
        return f"Emergency alert delivered via Twilio to {masked}: {delivered} sent."
    if sent:
        failed_detail = "; ".join(f"{result.channel} failed: {result.detail}" for result in failed)
        return f"Emergency alert partially delivered via Twilio to {masked}: {failed_detail}."
    if outcome.channels:
        failed_detail = "; ".join(f"{result.channel}: {result.detail}" for result in outcome.channels)
        return f"Twilio delivery failed ({failed_detail}). The alert was recorded in memory only."
    return "Twilio is configured but every notification channel is disabled. The alert was recorded in memory only."


def build_router(translator: DictionaryTranslator) -> APIRouter:
    router = APIRouter(prefix="/api/emergency", tags=["emergency"])
    log = EmergencyLog()

    @router.get("/phrases")
    def list_phrases(
        language: str = "en", notifier: TwilioNotifier = Depends(get_notifier)
    ) -> dict:
        status = notifier.status()
        if status["configured"]:
            note = (
                f"Alerts are delivered by Twilio to {status['to_masked']}. "
                "This notifies your configured contact — it still does not call public "
                "emergency services (112 / 911)."
            )
        else:
            note = NOT_REAL_DISPATCH_NOTE
        return {
            "language": language,
            "note": note,
            "notification": status,
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
    def provider_status(notifier: TwilioNotifier = Depends(get_notifier)) -> dict:
        """Report whether alerts are really delivered, and to which number."""
        return notifier.status()

    @router.post("/alert", response_model=AlertResponse)
    def raise_alert(
        payload: AlertRequest, notifier: TwilioNotifier = Depends(get_notifier)
    ) -> AlertResponse:
        phrase = find_phrase(payload.phrase_id)
        if phrase is None:
            raise HTTPException(404, f"Unknown emergency phrase id '{payload.phrase_id}'.")
        translated = _translate_phrase(translator, phrase.text, payload.language)
        outcome = notifier.notify(
            phrase_en=phrase.text,
            phrase_translated=translated,
            language=payload.language,
            context=payload.context_message,
        )
        log.record(payload.phrase_id, payload.language, delivery=outcome.delivery)
        return AlertResponse(
            acknowledged=True,
            phrase=phrase.text,
            translated=translated,
            note=_note_for(outcome),
            notification=_notification_status(outcome),
        )

    @router.get("/log")
    def recent_alerts() -> dict:
        return {
            "alerts": [
                {
                    "phrase_id": alert.phrase_id,
                    "language": alert.language,
                    "delivery": alert.delivery,
                    "timestamp": alert.timestamp,
                }
                for alert in log.recent()
            ]
        }

    return router
