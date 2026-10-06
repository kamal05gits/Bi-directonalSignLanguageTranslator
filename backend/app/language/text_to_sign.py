"""Text → sign direction: turn typed text into a fingerspelling plan.

This is the mirror image of the recognition pipeline. Instead of reading a
sign from the camera, it takes text a hearing person typed (or the message
the recognizer just built) and produces the ordered, step-by-step
fingerspelling sequence a signer can follow.

The rules are deliberately conservative, for the same reason the recognition
side never invents a label:

- Only characters the bundled letter models actually cover (a-z by default,
  intersected with the model's own label file) are returned as signable
  letters.
- Runs of whitespace collapse into a single ``space`` step, which is the
  pause between words rather than a sign of its own.
- Everything else - digits, punctuation, emoji, accented characters - is
  returned as an explicit ``unsupported`` step with a reason, never silently
  dropped and never mapped onto a lookalike letter. Digits get their own
  message because Indian Sign Language *does* have number signs; they are
  simply outside what these letter models cover.

The converter is pure Python (no TensorFlow, no model load), so the text →
sign direction keeps working on a deployment where the camera models are
unavailable, and it is unit-testable without any optional dependency.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field

MAX_TEXT_LENGTH = 240
DEFAULT_ALPHABET: tuple[str, ...] = tuple(chr(code) for code in range(ord("a"), ord("z") + 1))

LETTER = "letter"
SPACE = "space"
UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class SignStep:
    """One step of the fingerspelling sequence."""

    index: int
    kind: str
    character: str
    label: str
    hint: str
    word_index: int | None = None


@dataclass(frozen=True)
class SignSequence:
    """The full plan for signing a piece of text."""

    text: str
    normalized: str
    steps: list[SignStep] = field(default_factory=list)
    letter_count: int = 0
    word_count: int = 0
    unsupported: list[str] = field(default_factory=list)
    supported: bool = False
    message: str = ""

    @property
    def letters(self) -> list[str]:
        """Just the signable letters, in order (useful for playback/tests)."""
        return [step.label for step in self.steps if step.kind == LETTER]


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


class TextToSignConverter:
    """Convert typed text into an ordered fingerspelling sequence.

    ``letters`` is normally the bundled fingerspelling model's label list, so
    the guidance never promises a handshape the project does not actually
    recognize. When no usable label list is supplied (missing model files),
    it falls back to the a-z alphabet: the advice to a human signer is still
    correct even when the camera models are not deployed.
    """

    def __init__(self, letters: Sequence[str] | None = None) -> None:
        self.letters: list[str] = self._usable_letters(letters)

    @staticmethod
    def _usable_letters(letters: Sequence[str] | None) -> list[str]:
        candidates = {
            item.strip().lower()
            for item in (letters or ())
            if len(item.strip()) == 1 and item.strip().isalpha()
        }
        return sorted(candidates) or list(DEFAULT_ALPHABET)

    @property
    def alphabet(self) -> list[str]:
        return list(self.letters)

    def convert(self, text: str) -> SignSequence:
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text is limited to {MAX_TEXT_LENGTH} characters.")

        # NFKC folds compatibility forms (full-width letters, ligatures) onto
        # their plain equivalents. Accents are deliberately *not* stripped:
        # "é" is reported as unsupported instead of quietly becoming "e".
        normalized = " ".join(unicodedata.normalize("NFKC", text).lower().split())
        if not normalized:
            return SignSequence(
                text=text,
                normalized="",
                message="Nothing to sign yet — type some text above.",
            )

        steps: list[SignStep] = []
        unsupported: list[str] = []
        letter_count = 0
        word_index = 0
        at_word_start = True

        for character in normalized:
            if character == " ":
                steps.append(
                    SignStep(
                        index=len(steps),
                        kind=SPACE,
                        character=" ",
                        label="",
                        hint="Pause briefly — this is the gap between two words.",
                    )
                )
                at_word_start = True
                continue

            if at_word_start:
                word_index += 1
                at_word_start = False

            if character in self.letters:
                letter_count += 1
                steps.append(
                    SignStep(
                        index=len(steps),
                        kind=LETTER,
                        character=character,
                        label=character.upper(),
                        hint=f"Fingerspell the letter {character.upper()}.",
                        word_index=word_index,
                    )
                )
                continue

            if character not in unsupported:
                unsupported.append(character)
            steps.append(
                SignStep(
                    index=len(steps),
                    kind=UNSUPPORTED,
                    character=character,
                    label=character,
                    hint=self._unsupported_hint(character),
                    word_index=word_index,
                )
            )

        return SignSequence(
            text=text,
            normalized=normalized,
            steps=steps,
            letter_count=letter_count,
            word_count=word_index,
            unsupported=unsupported,
            supported=letter_count > 0 and not unsupported,
            message=self._summary(letter_count, word_index, unsupported),
        )

    @staticmethod
    def _unsupported_hint(character: str) -> str:
        if character.isdigit():
            return (
                f'"{character}" is a number. ISL has its own number signs, which these letter '
                "models do not cover — show the digit or say it instead."
            )
        if character.isalpha():
            return (
                f'"{character}" is outside the a-z alphabet these models cover — say or write it '
                "instead of guessing a handshape."
            )
        return f'"{character}" has no letter sign — say or write it instead.'

    @staticmethod
    def _summary(letter_count: int, word_count: int, unsupported: list[str]) -> str:
        summary = f"{_plural(letter_count, 'letter')} to sign across {_plural(word_count, 'word')}."
        if unsupported:
            listed = ", ".join(f'"{character}"' for character in unsupported)
            summary += (
                f" {_plural(len(unsupported), 'character')} cannot be fingerspelled ({listed}) — "
                "say or write those instead."
            )
        return summary
