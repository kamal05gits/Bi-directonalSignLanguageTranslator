"""Prototype emergency-phrase module.

This is explicitly **not** connected to real emergency dispatch: there is no
telephony, SMS gateway, or location service configured. What it gives a
deaf/hard-of-hearing user is fast, one-tap access to a small set of
pre-translated, high-value phrases that can be shown (large on-screen text)
and spoken aloud to a bystander - useful on its own - plus an in-memory audit
log of when an alert was raised, so a real provider (e.g. Twilio, a campus
security webhook) has one obvious place to be plugged in later
(:class:`EmergencyLog.record`).

Every response from the API built on top of this module says plainly that no
real service was contacted, matching the project's policy of never
fabricating a capability it does not have.
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


class EmergencyLog:
    """In-memory alert log. Not a substitute for a real dispatch integration."""

    def __init__(self, max_entries: int = 200) -> None:
        self._alerts: list[EmergencyAlert] = []
        self._max_entries = max_entries

    def record(self, phrase_id: str, language: str) -> EmergencyAlert:
        alert = EmergencyAlert(phrase_id=phrase_id, language=language)
        self._alerts.append(alert)
        del self._alerts[: -self._max_entries]
        return alert

    def recent(self, limit: int = 20) -> list[EmergencyAlert]:
        return self._alerts[-limit:]
