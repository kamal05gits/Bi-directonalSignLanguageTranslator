"""Unit tests for the text -> sign (fingerspelling plan) converter.

Dependency-free on purpose: the reverse direction must keep working - and
keep being tested - on a deployment without TensorFlow or FastAPI.
"""

import pytest
from app.language.text_to_sign import MAX_TEXT_LENGTH, TextToSignConverter


def kinds(sequence):
    return [step.kind for step in sequence.steps]


def test_letters_become_ordered_signable_steps():
    sequence = TextToSignConverter().convert("Hi")
    assert sequence.normalized == "hi"
    assert kinds(sequence) == ["letter", "letter"]
    assert sequence.letters == ["H", "I"]
    assert [step.index for step in sequence.steps] == [0, 1]
    assert sequence.letter_count == 2
    assert sequence.word_count == 1
    assert sequence.supported is True
    assert sequence.unsupported == []


def test_whitespace_runs_collapse_into_one_word_gap():
    sequence = TextToSignConverter().convert("  hi   sam \n")
    assert sequence.normalized == "hi sam"
    assert kinds(sequence) == ["letter", "letter", "space", "letter", "letter", "letter"]
    assert sequence.word_count == 2
    assert [step.word_index for step in sequence.steps] == [1, 1, None, 2, 2, 2]


def test_unsupported_characters_are_reported_not_dropped():
    sequence = TextToSignConverter().convert("a7!")
    assert kinds(sequence) == ["letter", "unsupported", "unsupported"]
    assert sequence.unsupported == ["7", "!"]
    assert sequence.supported is False
    assert "7" in sequence.steps[1].hint and "number" in sequence.steps[1].hint.lower()
    assert "cannot be fingerspelled" in sequence.message


def test_accented_letters_are_not_silently_folded_onto_a_lookalike():
    sequence = TextToSignConverter().convert("café")
    assert kinds(sequence) == ["letter", "letter", "letter", "unsupported"]
    assert sequence.unsupported == ["é"]


def test_empty_text_returns_an_empty_plan_with_a_message():
    sequence = TextToSignConverter().convert("   ")
    assert sequence.steps == []
    assert sequence.letter_count == 0
    assert sequence.supported is False
    assert "Nothing to sign" in sequence.message


def test_text_longer_than_the_limit_is_rejected():
    with pytest.raises(ValueError):
        TextToSignConverter().convert("a" * (MAX_TEXT_LENGTH + 1))


def test_alphabet_follows_the_model_labels_when_they_are_available():
    converter = TextToSignConverter(["A", "b", "c"])
    assert converter.alphabet == ["a", "b", "c"]
    sequence = converter.convert("abz")
    assert kinds(sequence) == ["letter", "letter", "unsupported"]
    assert sequence.unsupported == ["z"]


def test_alphabet_falls_back_to_a_z_without_usable_labels():
    # Missing model files (labels == []) must not break the guidance, and
    # non-letter labels like "space"/"del" are not fingerspellable letters.
    assert TextToSignConverter([]).alphabet == [chr(code) for code in range(ord("a"), ord("z") + 1)]
    assert TextToSignConverter(["space", "del"]).alphabet[0] == "a"


def test_summary_counts_letters_and_words():
    sequence = TextToSignConverter().convert("hi sam")
    assert sequence.message == "5 letters to sign across 2 words."


def test_every_step_carries_a_human_readable_hint():
    sequence = TextToSignConverter().convert("hi sam!")
    assert all(step.hint for step in sequence.steps)
    assert sequence.steps[2].kind == "space"
    assert "Pause" in sequence.steps[2].hint
