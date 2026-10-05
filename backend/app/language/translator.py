from dataclasses import dataclass

@dataclass
class TranslationResult:
    success: bool
    text: str
    message: str = ""

class DictionaryTranslator:
    DICTIONARIES = {"ta": {"hello": "வணக்கம்", "thank": "நன்றி", "you": "நீங்கள்", "good": "நல்ல", "morning": "காலை"}}
    def translate(self, text, language):
        dictionary = self.DICTIONARIES.get(language, {})
        words = text.lower().split(); unknown = [w for w in words if w not in dictionary]
        if unknown: return TranslationResult(False, "", "Unknown words: " + ", ".join(unknown))
        return TranslationResult(True, " ".join(dictionary[w] for w in words))
