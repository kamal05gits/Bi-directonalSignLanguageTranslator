"""Emergency-phrase module.

Gives a deaf/hard-of-hearing user fast, one-tap access to a small set of
pre-translated, high-value phrases that can be shown (large on-screen text)
and spoken aloud to a bystander, plus an in-memory audit log of when an
alert was raised.

Real dispatch is optional and handled by :mod:`app.services.notifications`
(Twilio SMS + voice call) - see :class:`app.services.notifications.TwilioNotifier`.
When Twilio is not configured (no ``TWILIO_*`` environment variables), this
stays a prototype exactly as before: no telephony, SMS gateway, or location
service is contacted, and the API says so plainly rather than fabricating a
capability it does not have.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EmergencyPhrase:
    id: str
    text: str  # English source text; also a translator.PHRASES lookup key.


PHRASES: list[EmergencyPhrase] = [
    EmergencyPhrase("help", "I need help"),
    EmergencyPhrase("ambulance", "call an ambulance"),
    EmergencyPhrase("police", "call the police"),
    EmergencyPhrase("doctor", "I need a doctor"),
    EmergencyPhrase("deaf", "I am deaf"),
    EmergencyPhrase("fire", "there is a fire"),
    EmergencyPhrase("lost", "I am lost"),
    EmergencyPhrase("hospital", "where is the hospital"),
]

_PHRASES_BY_ID = {phrase.id: phrase for phrase in PHRASES}


def find_phrase(phrase_id: str) -> EmergencyPhrase | None:
    return _PHRASES_BY_ID.get(phrase_id)


@dataclass
class EmergencyAlert:
    phrase_id: str
    language: str
    timestamp: float = field(default_factory=time.time)
    dispatched: bool = False
    sms_sid: str | None = None
    call_sid: str | None = None
    dispatch_errors: list[str] = field(default_factory=list)


class EmergencyLog:
    """In-memory alert log, including whether a real Twilio dispatch happened."""

    def __init__(self, max_entries: int = 200) -> None:
        self._alerts: list[EmergencyAlert] = []
        self._max_entries = max_entries

    def record(
        self,
        phrase_id: str,
        language: str,
        dispatched: bool = False,
        sms_sid: str | None = None,
        call_sid: str | None = None,
        dispatch_errors: list[str] | None = None,
    ) -> EmergencyAlert:
        alert = EmergencyAlert(
            phrase_id=phrase_id,
            language=language,
            dispatched=dispatched,
            sms_sid=sms_sid,
            call_sid=call_sid,
            dispatch_errors=dispatch_errors or [],
        )
        self._alerts.append(alert)
        del self._alerts[: -self._max_entries]
        return alert

    def recent(self, limit: int = 20) -> list[EmergencyAlert]:
        return self._alerts[-limit:]
