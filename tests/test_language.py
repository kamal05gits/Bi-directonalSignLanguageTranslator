from app.language.sentence_processor import SentenceProcessor
from app.language.translator import DictionaryTranslator


def test_sentence_processor_preserves_tokens_and_punctuation():
    processor = SentenceProcessor()
    processor.append_token("hello")
    processor.append_token("thank_you")
    processor.append_punctuation(".")
    assert processor.text == "hello thank you."
    assert processor.original_tokens == ["hello", "thank_you", "period"]


def test_translation_refuses_unknown_words():
    result = DictionaryTranslator().translate("hello unknown", "ta")
    assert not result.success
    assert "unknown" in result.message


def test_character_mode_concatenates_fingerspelling_letters():
    processor = SentenceProcessor(token_mode="character")
    processor.append_token("h")
    processor.append_token("i")
    processor.insert_space()
    processor.append_token("a")
    assert processor.text == "hi a"
