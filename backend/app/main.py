"""FastAPI application for the bidirectional sign-language translator.

The heavy TensorFlow model is loaded lazily, so health checks remain responsive
while a Render instance starts. The same service hosts the static web client and
its JSON API, avoiding CORS and cross-origin camera issues.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from .services.alphabet_predictor import AlphabetPredictor, Prediction

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
MODEL = Path(os.getenv("ALPHABET_MODEL_PATH", ROOT / "backend/models/alphabet/isl_alphabet_model.keras"))
LABELS = Path(os.getenv("ALPHABET_LABELS_PATH", ROOT / "backend/models/alphabet/alphabet_labels.json"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 5 * 1024 * 1024))

app = FastAPI(
    title="Bidirectional Sign Language Translator",
    description="Indian Sign Language fingerspelling recognition API",
    version="1.0.0",
)
predictor = AlphabetPredictor(MODEL, LABELS)


class HealthResponse(BaseModel):
    status: str
    model_available: bool
    model_loaded: bool


class PredictionResponse(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    accepted: bool
    top_predictions: list[Prediction]


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_available=MODEL.is_file() and LABELS.is_file(),
        model_loaded=predictor.loaded,
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: Annotated[UploadFile, File(description="JPEG, PNG, or WebP hand image")]) -> PredictionResponse:
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Please upload a JPEG, PNG, or WebP image.")

    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image exceeds the 5 MB upload limit.")

    try:
        image = Image.open(io.BytesIO(payload))
        image = ImageOps.exif_transpose(image).convert("RGB")
        result = predictor.predict(image)
    except (OSError, ValueError) as exc:
        raise HTTPException(400, f"Invalid image: {exc}") from exc
    except RuntimeError as exc:
        LOGGER.exception("Prediction failed")
        raise HTTPException(503, str(exc)) from exc

    threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
    return PredictionResponse(
        label=result[0].label,
        confidence=result[0].confidence,
        accepted=result[0].confidence >= threshold,
        top_predictions=result,
    )


@app.get("/api/info")
def info() -> dict[str, object]:
    return {
        "recognition": "single-frame ISL alphabet fingerspelling",
        "labels": predictor.labels,
        "input_guidance": "Keep one hand centered in the guide with a plain, well-lit background.",
    }


# Mount assets after API routes so /api is never swallowed by the static app.
app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def spa(path: str = "") -> FileResponse:
    requested = (FRONTEND / path).resolve()
    if path and requested.is_relative_to(FRONTEND.resolve()) and requested.is_file():
        return FileResponse(requested)
    return FileResponse(FRONTEND / "index.html")
