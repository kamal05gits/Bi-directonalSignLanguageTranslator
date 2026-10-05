class SentenceProcessor:
    def __init__(self, token_mode="word"):
        self.token_mode = token_mode; self.original_tokens = []; self._text = ""
    @property
    def text(self): return self._text
    def append_token(self, token):
        self.original_tokens.append(token)
        display = token.replace("_", " ")
        if self.token_mode == "character": self._text += display
        else: self._text += (" " if self._text and not self._text.endswith(" ") else "") + display
    def append_punctuation(self, mark):
        names = {".": "period", ",": "comma", "?": "question"}
        self.original_tokens.append(names.get(mark, mark)); self._text = self._text.rstrip() + mark
    def insert_space(self):
        if self._text and not self._text.endswith(" "): self._text += " "
