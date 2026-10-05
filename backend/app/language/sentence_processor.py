"""Converts a stream of recognized sign tokens into readable text.

Tokens are kept as an ordered list of "pieces" (a token, a punctuation mark,
or an inserted space) rather than only the flattened string. That makes the
module genuinely support manual correction (:meth:`backspace`) by replaying
every remaining piece instead of guessing how to edit a plain string, which
is what the continuous-recognition API (``app.routes.continuous``) relies on.
"""

from __future__ import annotations

PUNCTUATION_NAMES = {".": "period", ",": "comma", "?": "question", "!": "exclamation"}


class SentenceProcessor:
    """Builds an editable sentence from ``word`` or ``character`` tokens."""

    def __init__(self, token_mode: str = "word") -> None:
        self.token_mode = token_mode
        self._pieces: list[dict] = []
        self._text = ""

    @property
    def text(self) -> str:
        return self._text

    @property
    def original_tokens(self) -> list[str]:
        """The raw token trace (e.g. ``["hello", "thank_you", "period"]``)."""
        tokens = []
        for piece in self._pieces:
            if piece["type"] == "token":
                tokens.append(piece["token"])
            elif piece["type"] == "punct":
                tokens.append(piece["name"])
        return tokens

    def append_token(self, token: str) -> None:
        self._pieces.append({"type": "token", "token": token})
        self._rebuild()

    def append_punctuation(self, mark: str) -> None:
        self._pieces.append({"type": "punct", "mark": mark, "name": PUNCTUATION_NAMES.get(mark, mark)})
        self._rebuild()

    def insert_space(self) -> None:
        self._pieces.append({"type": "space"})
        self._rebuild()

    def backspace(self) -> bool:
        """Remove the most recently added piece. Returns False if already empty."""
        if not self._pieces:
            return False
        self._pieces.pop()
        self._rebuild()
        return True

    def clear(self) -> None:
        self._pieces = []
        self._text = ""

    def _rebuild(self) -> None:
        text = ""
        for piece in self._pieces:
            if piece["type"] == "token":
                display = piece["token"].replace("_", " ")
                if self.token_mode == "character":
                    text += display
                else:
                    text += (" " if text and not text.endswith(" ") else "") + display
            elif piece["type"] == "punct":
                text = text.rstrip() + piece["mark"]
            elif piece["type"] == "space":
                if text and not text.endswith(" "):
                    text += " "
        self._text = text
