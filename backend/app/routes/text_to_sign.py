"""Text → sign endpoints (the reverse direction of the translator).

``POST /api/text-to-sign`` takes typed text and returns the ordered
fingerspelling plan produced by
:class:`app.language.text_to_sign.TextToSignConverter`: one step per
character, with letters, word pauses, and explicitly-flagged characters that
have no letter sign. ``GET /api/text-to-sign/alphabet`` reports which letters
the deployment can actually offer, so the UI can describe its own coverage
instead of assuming a-z.

Nothing here loads a model, so this direction keeps working (and keeps being
testable) on a deployment without TensorFlow.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..language.text_to_sign import MAX_TEXT_LENGTH, TextToSignConverter


class TextToSignRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)


class SignStepItem(BaseModel):
    index: int
    kind: str
    character: str
    label: str
    hint: str
    word_index: int | None = None


class TextToSignResponse(BaseModel):
    text: str
    normalized: str
    supported: bool
    letter_count: int
    word_count: int
    unsupported: list[str]
    message: str
    steps: list[SignStepItem]


class AlphabetResponse(BaseModel):
    letters: list[str]
    count: int
    note: str


def build_router(converter: TextToSignConverter) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["text-to-sign"])

    @router.get("/text-to-sign/alphabet", response_model=AlphabetResponse)
    def alphabet() -> AlphabetResponse:
        return AlphabetResponse(
            letters=converter.alphabet,
            count=len(converter.alphabet),
            note=(
                "Fingerspelling guidance only — an ordered letter sequence for a human signer, "
                "not generated sign-language video. Characters outside this alphabet are reported "
                "as unsupported instead of being mapped onto a lookalike sign."
            ),
        )

    @router.post("/text-to-sign", response_model=TextToSignResponse)
    def text_to_sign(payload: TextToSignRequest) -> TextToSignResponse:
        try:
            sequence = converter.convert(payload.text)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return TextToSignResponse(
            text=sequence.text,
            normalized=sequence.normalized,
            supported=sequence.supported,
            letter_count=sequence.letter_count,
            word_count=sequence.word_count,
            unsupported=sequence.unsupported,
            message=sequence.message,
            steps=[
                SignStepItem(
                    index=step.index,
                    kind=step.kind,
                    character=step.character,
                    label=step.label,
                    hint=step.hint,
                    word_index=step.word_index,
                )
                for step in sequence.steps
            ],
        )

    return router
