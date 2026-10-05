"""Emergency-phrase module.

Phrases give a deaf/hard-of-hearing user fast, one-tap access to a small set
of pre-translated, high-value texts that can be shown (large on-screen text)
and spoken aloud to a bystander - useful on its own - plus an audit log of
when an alert was raised and whether it was delivered.

Real delivery is wired through :class:`app.services.twilio_notifier.TwilioNotifier`
(SMS + voice call to a configured number). When Twilio is not configured the
feature stays a prototype, and every API response says so plainly - matching
the project's policy of never fabricating a capability it does not have.
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
    delivery: str = "none"  # e.g. "sms+call", "sms", "none"
    timestamp: float = field(default_factory=time.time)


class EmergencyLog:
    """In-memory alert log. Not a substitute for a real dispatch integration."""

    def __init__(self, max_entries: int = 200) -> None:
        self._alerts: list[EmergencyAlert] = []
        self._max_entries = max_entries

    def record(self, phrase_id: str, language: str, delivery: str = "none") -> EmergencyAlert:
        alert = EmergencyAlert(phrase_id=phrase_id, language=language, delivery=delivery)
        self._alerts.append(alert)
        del self._alerts[: -self._max_entries]
        return alert

    def recent(self, limit: int = 20) -> list[EmergencyAlert]:
        return self._alerts[-limit:]
