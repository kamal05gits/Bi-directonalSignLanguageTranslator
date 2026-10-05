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


def test_backspace_undoes_the_last_piece_and_can_be_repeated():
    processor = SentenceProcessor(token_mode="character")
    processor.append_token("h")
    processor.append_token("i")
    processor.append_punctuation(".")
    assert processor.text == "hi."
    assert processor.backspace() is True
    assert processor.text == "hi"
    assert processor.original_tokens == ["h", "i"]
    assert processor.backspace() is True
    assert processor.text == "h"
    assert processor.backspace() is True
    assert processor.text == ""
    assert processor.backspace() is False  # already empty


def test_clear_resets_tokens_and_text():
    processor = SentenceProcessor()
    processor.append_token("hello")
    processor.clear()
    assert processor.text == ""
    assert processor.original_tokens == []


def test_translation_resolves_whole_phrases_before_word_by_word():
    result = DictionaryTranslator().translate("Thank You", "hi")
    assert result.success
    assert result.text == "धन्यवाद"


def test_translation_supports_multiple_languages_and_reports_unresolved_words():
    translator = DictionaryTranslator()
    tamil = translator.translate("hello water", "ta")
    assert tamil.success
    assert tamil.text == "வணக்கம் தண்ணீர்"

    bad_language = translator.translate("hello", "fr")
    assert not bad_language.success
    assert "Unsupported language" in bad_language.message

    unresolved = translator.translate("hello spaceship", "ta")
    assert not unresolved.success
    assert unresolved.unresolved == ["spaceship"]
