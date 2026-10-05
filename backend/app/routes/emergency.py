"""Emergency-phrase API.

Explicitly a prototype feature: it returns pre-translated, high-value
phrases for fast on-screen/spoken use in a crisis, and keeps an in-memory
log of raised alerts. It does **not** call real emergency services, SMS, or
telephony - every response says so plainly. Wiring a real provider is a
single extension point: see :class:`app.services.emergency.EmergencyLog`.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..language.translator import DictionaryTranslator
from ..services.emergency import PHRASES, EmergencyLog, find_phrase

NOT_REAL_DISPATCH_NOTE = (
    "Prototype only: no real emergency service, SMS, or phone call was placed. "
    "This alert was recorded in memory for demo purposes only."
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


def build_router(translator: DictionaryTranslator) -> APIRouter:
    router = APIRouter(prefix="/api/emergency", tags=["emergency"])
    log = EmergencyLog()

    @router.get("/phrases")
    def list_phrases(language: str = "en") -> dict:
        return {
            "language": language,
            "note": NOT_REAL_DISPATCH_NOTE,
            "phrases": [
                {
                    "id": phrase.id,
                    "text": phrase.text,
                    "translated": _translate_phrase(translator, phrase.text, language),
                }
                for phrase in PHRASES
            ],
        }

    @router.post("/alert", response_model=AlertResponse)
    def raise_alert(payload: AlertRequest) -> AlertResponse:
        phrase = find_phrase(payload.phrase_id)
        if phrase is None:
            raise HTTPException(404, f"Unknown emergency phrase id '{payload.phrase_id}'.")
        log.record(payload.phrase_id, payload.language)
        return AlertResponse(
            acknowledged=True,
            phrase=phrase.text,
            translated=_translate_phrase(translator, phrase.text, payload.language),
            note=NOT_REAL_DISPATCH_NOTE,
        )

    @router.get("/log")
    def recent_alerts() -> dict:
        return {
            "alerts": [
                {"phrase_id": alert.phrase_id, "language": alert.language, "timestamp": alert.timestamp}
                for alert in log.recent()
            ]
        }

    return router
