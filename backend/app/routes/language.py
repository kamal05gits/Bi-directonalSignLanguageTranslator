"""Translation and supported-language endpoints (the "Multilingual Output" module)."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..language.translator import DictionaryTranslator


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=240)
    language: str


class TranslateResponse(BaseModel):
    success: bool
    original: str
    language: str
    translated: str
    message: str = ""
    unresolved: list[str] = Field(default_factory=list)


def build_router(translator: DictionaryTranslator) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["language"])

    @router.get("/languages")
    def list_languages() -> dict:
        return {
            "languages": [
                {"code": code, "name": name} for code, name in translator.LANGUAGES.items()
            ]
        }

    @router.post("/translate", response_model=TranslateResponse)
    def translate(payload: TranslateRequest) -> TranslateResponse:
        result = translator.translate(payload.text, payload.language)
        return TranslateResponse(
            success=result.success,
            original=payload.text,
            language=payload.language,
            translated=result.text,
            message=result.message,
            unresolved=result.unresolved,
        )

    return router
