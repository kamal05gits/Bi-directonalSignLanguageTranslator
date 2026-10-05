"""Dictionary-based multilingual translation for recognized ISL text.

SignBridge is a single free-tier web service with no paid translation API
key, so this module translates with a curated bundled dictionary instead of
a live network call to a third-party translation service. That keeps it
deterministic, free, and fully offline-capable - and consistent with the
rest of the project's "never fabricate a result" principle: unresolved words
are reported explicitly (``TranslationResult.unresolved``) instead of being
silently echoed back untranslated or guessed at.

Whole-phrase entries are checked before the per-word fallback so common,
grammatically correct phrases (including the emergency phrases in
``app.services.emergency``) translate naturally instead of word-for-word.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TranslationResult:
    success: bool
    text: str
    message: str = ""
    unresolved: list[str] = field(default_factory=list)


class DictionaryTranslator:
    """Translate English text into a supported target language."""

    LANGUAGES: dict[str, str] = {
        "ta": "Tamil (தமிழ்)",
        "hi": "Hindi (हिन्दी)",
    }

    PHRASES: dict[str, dict[str, str]] = {
        "ta": {
            "thank you": "நன்றி",
            "good morning": "காலை வணக்கம்",
            "i need help": "எனக்கு உதவி தேவை",
            "call an ambulance": "ஆம்புலன்ஸை அழைக்கவும்",
            "call the police": "காவல்துறையை அழைக்கவும்",
            "i am deaf": "நான் காது கேளாதவன்",
            "where is the hospital": "மருத்துவமனை எங்கே",
            "i need a doctor": "எனக்கு மருத்துவர் தேவை",
            "there is a fire": "தீ விபத்து",
            "i am lost": "நான் வழி தவறிவிட்டேன்",
        },
        "hi": {
            "thank you": "धन्यवाद",
            "good morning": "सुप्रभात",
            "i need help": "मुझे मदद चाहिए",
            "call an ambulance": "एम्बुलेंस बुलाओ",
            "call the police": "पुलिस को बुलाओ",
            "i am deaf": "मैं बहरा हूँ",
            "where is the hospital": "अस्पताल कहाँ है",
            "i need a doctor": "मुझे डॉक्टर चाहिए",
            "there is a fire": "आग लगी है",
            "i am lost": "मैं खो गया हूँ",
        },
    }

    WORDS: dict[str, dict[str, str]] = {
        "ta": {
            "hello": "வணக்கம்", "thank": "நன்றி", "you": "நீங்கள்", "good": "நல்ல",
            "morning": "காலை", "afternoon": "மதியம்", "evening": "மாலை", "night": "இரவு",
            "please": "தயவுசெய்து", "sorry": "மன்னிக்கவும்", "yes": "ஆம்", "no": "இல்லை",
            "help": "உதவி", "water": "தண்ணீர்", "food": "உணவு", "doctor": "மருத்துவர்",
            "police": "காவல்துறை", "fire": "தீ", "ambulance": "ஆம்புலன்ஸ்", "emergency": "அவசரநிலை",
            "call": "அழை", "need": "தேவை", "home": "வீடு", "school": "பள்ளி",
            "friend": "நண்பர்", "family": "குடும்பம்", "love": "அன்பு", "name": "பெயர்",
            "hospital": "மருத்துவமனை", "deaf": "காது கேளாதவர்", "lost": "தொலைந்துவிட்டேன்",
            "where": "எங்கே", "what": "என்ன", "my": "என்னுடைய", "i": "நான்", "is": "ஆகும்",
            "one": "ஒன்று", "two": "இரண்டு", "three": "மூன்று", "four": "நான்கு", "five": "ஐந்து",
            "six": "ஆறு", "seven": "ஏழு", "eight": "எட்டு", "nine": "ஒன்பது", "ten": "பத்து",
        },
        "hi": {
            "hello": "नमस्ते", "thank": "धन्यवाद", "you": "आप", "good": "अच्छा",
            "morning": "सुबह", "afternoon": "दोपहर", "evening": "शाम", "night": "रात",
            "please": "कृपया", "sorry": "माफ़ करें", "yes": "हाँ", "no": "नहीं",
            "help": "मदद", "water": "पानी", "food": "खाना", "doctor": "डॉक्टर",
            "police": "पुलिस", "fire": "आग", "ambulance": "एम्बुलेंस", "emergency": "आपातकाल",
            "call": "बुलाओ", "need": "चाहिए", "home": "घर", "school": "स्कूल",
            "friend": "दोस्त", "family": "परिवार", "love": "प्यार", "name": "नाम",
            "hospital": "अस्पताल", "deaf": "बहरा", "lost": "खो गया",
            "where": "कहाँ", "what": "क्या", "my": "मेरा", "i": "मैं", "is": "है",
            "one": "एक", "two": "दो", "three": "तीन", "four": "चार", "five": "पांच",
            "six": "छह", "seven": "सात", "eight": "आठ", "nine": "नौ", "ten": "दस",
        },
    }

    def translate(self, text: str, language: str) -> TranslationResult:
        if language not in self.LANGUAGES:
            return TranslationResult(False, "", f"Unsupported language '{language}'.")
        normalized = " ".join(text.strip().lower().split())
        if not normalized:
            return TranslationResult(False, "", "Nothing to translate.")

        phrase_match = self.PHRASES.get(language, {}).get(normalized)
        if phrase_match is not None:
            return TranslationResult(True, phrase_match)

        dictionary = self.WORDS.get(language, {})
        words = normalized.split(" ")
        unknown = [word for word in words if word not in dictionary]
        if unknown:
            return TranslationResult(
                False, "", "Unknown words: " + ", ".join(unknown), unresolved=unknown
            )
        return TranslationResult(True, " ".join(dictionary[word] for word in words))
