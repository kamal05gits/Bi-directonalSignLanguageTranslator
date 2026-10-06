"""FastAPI application for the bidirectional sign-language translator.

Three recognition models are served behind one API, and can be used one at
a time or combined in a single soft-voting ensemble request:

- ``alphabet``: single-frame 64x64 photo CNN (bundled, always deployable), or
  the optional Roboflow ISL object detector when ``ROBOFLOW_API_KEY`` is set.
- ``fingerspelling``: MLP over a 126-value MediaPipe hand-landmark vector
  extracted in the browser (bundled, always deployable).
- ``word``: CISLR word classifier over I3D video features. It needs the Git
  LFS checkpoint plus optional PyTorch/OpenCV dependencies, and reports itself
  honestly as unavailable when those are missing.
- ``combined``: ``POST /api/predict/combined`` runs every provided input
  through its model and merges the results (see ``app.services.ensemble``).

Emergency alerts are delivered for real when Twilio is configured via
environment variables (``app.services.twilio_notifier``), and stay a
prototype that says so plainly when it is not.

The heavy models are loaded lazily, so health checks remain responsive while a
Render instance starts. The same service hosts the static web client and its
JSON API, avoiding CORS and cross-origin camera issues.
"""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from .language.translator import DictionaryTranslator
from .routes import continuous as continuous_routes
from .routes import emergency as emergency_routes
from .routes import language as language_routes
from .services.alphabet_predictor import AlphabetPredictor
from .services.ensemble import EnsembleError, ModelSource, combine, not_run
from .services.fingerspelling_predictor import FEATURE_DIM, FingerspellingPredictor
from .services.keras_classifier import Prediction
from .services.roboflow_predictor import RoboflowPredictor
from .services.twilio_notifier import TwilioConfig, TwilioNotifier
from .services.word_predictor import WordPredictor

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
MODEL = Path(os.getenv("ALPHABET_MODEL_PATH", ROOT / "backend/models/alphabet/isl_alphabet_model.keras"))
LABELS = Path(os.getenv("ALPHABET_LABELS_PATH", ROOT / "backend/models/alphabet/alphabet_labels.json"))
FINGERSPELLING_MODEL = Path(os.getenv("FINGERSPELLING_MODEL_PATH", ROOT / "backend/models/fingerspelling/isl_fingerspelling_model.keras"))
FINGERSPELLING_LABELS = Path(os.getenv("FINGERSPELLING_LABELS_PATH", ROOT / "backend/models/fingerspelling/isl_fingerspelling_labels.json"))
WORD_MODEL = Path(os.getenv("WORD_MODEL_PATH", ROOT / "backend/models/cislr/CISLR_MODEL.keras"))
WORD_LABELS = Path(os.getenv("WORD_LABELS_PATH", ROOT / "backend/models/cislr/CISLR_LABELS.json"))
WORD_NORMALIZATION = Path(os.getenv("WORD_NORMALIZATION_PATH", ROOT / "backend/models/cislr/CISLR_NORMALIZATION.npz"))
I3D_WEIGHTS = Path(os.getenv(
    "I3D_WEIGHTS_PATH",
    ROOT / "backend/i3d/weights/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt",
))
I3D_CODE_DIR = Path(os.getenv("I3D_CODE_DIR", ROOT / "backend/i3d"))
I3D_WEIGHTS_URL = os.getenv(
    "I3D_WEIGHTS_URL",
    "https://media.githubusercontent.com/media/kamal05gits/Bi-directonalSignLanguageTranslator/"
    "main/backend/i3d/weights/asl2000/"
    "FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt",
)
I3D_WEIGHTS_SHA256 = os.getenv(
    "I3D_WEIGHTS_SHA256", "243a19e6deef3becffbfc5b7dd8adb32916c8bee482565ca243c082584732620"
)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 5 * 1024 * 1024))
MAX_VIDEO_BYTES = int(os.getenv("MAX_VIDEO_BYTES", 32 * 1024 * 1024))
VIDEO_SUFFIXES = {".webm", ".mp4", ".mov", ".m4v", ".avi"}
VIDEO_CONTENT_TYPES = {"video/webm", "video/mp4", "video/quicktime", "video/x-msvideo", "application/octet-stream"}

app = FastAPI(
    title="Bidirectional Sign Language Translator",
    description="Indian Sign Language recognition API: alphabet photo model, hand-landmark fingerspelling model, CISLR word video model, a soft-voting combined ensemble, and Twilio-backed emergency alerts",
    version="2.1.0",
)
def _env_float(env_var: str, default: str) -> float:
    try:
        return float(os.getenv(env_var, default))
    except ValueError:
        return float(default)


alphabet_predictor = AlphabetPredictor(MODEL, LABELS)
# The bundled fingerspelling MLP is heavily overconfident (it answers ~100%
# even for random input), which is what buried correct signs at a few percent
# in the suggestions. Temperature scaling softens its softmax to honest
# probabilities, and mirror TTA rescues signs made with the opposite hand /
# flipped MediaPipe handedness. Both are tunable without a redeploy.
fingerspelling_predictor = FingerspellingPredictor(
    FINGERSPELLING_MODEL,
    FINGERSPELLING_LABELS,
    temperature=_env_float("FINGERSPELLING_TEMPERATURE", "2.5"),
    mirror_tta=os.getenv("FINGERSPELLING_MIRROR_TTA", "1") not in {"0", "false", "False", ""},
    mirror_margin=_env_float("FINGERSPELLING_MIRROR_MARGIN", "1.25"),
)
word_predictor = WordPredictor(
    WORD_MODEL,
    WORD_LABELS,
    WORD_NORMALIZATION,
    I3D_WEIGHTS,
    I3D_CODE_DIR,
    checkpoint_url=(
        I3D_WEIGHTS_URL
        if os.getenv("WORD_AUTO_DOWNLOAD", "1") not in {"0", "false", "False", ""}
        else None
    ),
    checkpoint_sha256=I3D_WEIGHTS_SHA256,
)
# Optional hosted detector. When ROBOFLOW_API_KEY is absent, the bundled
# alphabet classifier remains the image provider and the app works offline.
roboflow_predictor = RoboflowPredictor()
translator = DictionaryTranslator()
twilio_notifier = TwilioNotifier(TwilioConfig.from_env())


class ModelHealth(BaseModel):
    available: bool
    loaded: bool
    detail: str


class HealthResponse(BaseModel):
    status: str
    model_available: bool
    model_loaded: bool
    models: dict[str, ModelHealth]
    providers: dict[str, ModelHealth] = Field(default_factory=dict)


class PredictionItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    label: str
    confidence: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    accepted: bool
    top_predictions: list[PredictionItem]


class CombinedSourceItem(BaseModel):
    model: str
    ran: bool
    ok: bool
    label: str | None = None
    confidence: float | None = Field(None, ge=0, le=1)
    detail: str


class CombinedPredictionResponse(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    accepted: bool
    agreement: bool | None
    method: str
    word: PredictionItem | None
    sources: list[CombinedSourceItem]
    top_predictions: list[PredictionItem]


MAX_LANDMARK_FRAMES = 10


class LandmarkRequest(BaseModel):
    """One landmark vector, or a short burst of them for the same sign.

    ``frames`` lets the client send several samples captured milliseconds
    apart; the server averages the model's probability distributions across
    them, which is markedly more reliable than trusting a single frame.
    """

    landmarks: list[float] | None = Field(None, min_length=FEATURE_DIM, max_length=FEATURE_DIM)
    frames: list[list[float]] | None = Field(None, min_length=1, max_length=MAX_LANDMARK_FRAMES)

    def all_frames(self) -> list[list[float]]:
        frames = list(self.frames or [])
        if self.landmarks:
            frames.append(self.landmarks)
        return frames


def _confidence_threshold(env_var: str, default: str) -> float:
    try:
        return float(os.getenv(env_var, default))
    except ValueError:
        return float(default)


def _is_accepted(results: list[Prediction], threshold: float) -> bool:
    """Confidence- and margin-aware acceptance.

    A prediction is accepted when it clears the absolute threshold, or when
    it is moderately confident *and* clearly ahead of the runner-up. The
    margin rule rescues correct-but-softly-scored predictions (expected now
    that temperature calibration spreads mass over similar-looking signs)
    without accepting genuinely ambiguous ones, where the top two scores
    are close.
    """
    top = results[0].confidence
    if top >= threshold:
        return True
    runner_up = results[1].confidence if len(results) > 1 else 0.0
    return top >= threshold * 0.6 and top >= runner_up * 2.0


def _response(results: list[Prediction], threshold: float) -> PredictionResponse:
    return PredictionResponse(
        label=results[0].label,
        confidence=results[0].confidence,
        accepted=_is_accepted(results, threshold),
        top_predictions=results,
    )


def _roboflow_available() -> bool:
    """Return whether the optional hosted detector is configured and usable."""
    return roboflow_predictor.availability()[0]


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    word_available, word_detail = word_predictor.availability()
    roboflow_available, roboflow_detail = roboflow_predictor.availability()
    local_alphabet_available = alphabet_predictor.available
    alphabet_available = roboflow_available or local_alphabet_available
    if roboflow_available:
        alphabet_detail = roboflow_detail
        alphabet_loaded = roboflow_predictor.loaded
    elif local_alphabet_available:
        alphabet_detail = "ready (bundled model)"
        alphabet_loaded = alphabet_predictor.loaded
    else:
        alphabet_detail = "Alphabet model files are missing."
        alphabet_loaded = False
    models = {
        "alphabet": ModelHealth(
            available=alphabet_available,
            loaded=alphabet_loaded,
            detail=alphabet_detail,
        ),
        "fingerspelling": ModelHealth(
            available=fingerspelling_predictor.available,
            loaded=fingerspelling_predictor.loaded,
            detail="ready" if fingerspelling_predictor.available else "Fingerspelling model files are missing.",
        ),
        "word": ModelHealth(available=word_available, loaded=word_predictor.loaded, detail=word_detail),
    }
    alphabet_health = models["alphabet"]
    return HealthResponse(
        status="ok",
        model_available=alphabet_health.available,
        model_loaded=alphabet_health.loaded,
        models=models,
        providers={
            "roboflow": ModelHealth(
                available=roboflow_available,
                loaded=roboflow_predictor.loaded,
                detail=roboflow_detail,
            )
        },
    )


def _decode_image(file: UploadFile, payload: bytes) -> Image.Image:
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Please upload a JPEG, PNG, or WebP image.")
    try:
        image = Image.open(io.BytesIO(payload))
        return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, ValueError) as exc:
        raise HTTPException(400, f"Invalid image: {exc}") from exc


async def _read_upload(file: UploadFile, limit: int, error: str) -> bytes:
    payload = await file.read(limit + 1)
    if len(payload) > limit:
        raise HTTPException(413, error)
    return payload


async def _predict_image(
    file: UploadFile, payload: bytes, *, force_roboflow: bool = False
) -> PredictionResponse:
    image = _decode_image(file, payload)
    use_roboflow = force_roboflow or _roboflow_available()
    try:
        if use_roboflow:
            result = await run_in_threadpool(roboflow_predictor.predict, image, 5)
        else:
            result = alphabet_predictor.predict(image, top_k=5)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        LOGGER.exception("Image prediction failed")
        raise HTTPException(503, str(exc)) from exc
    return _response(result, _confidence_threshold("CONFIDENCE_THRESHOLD", "0.60"))


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: Annotated[UploadFile, File(description="JPEG, PNG, or WebP hand image")]) -> PredictionResponse:
    """Backward-compatible alias of /api/predict/alphabet."""
    payload = await _read_upload(file, MAX_UPLOAD_BYTES, "Image exceeds the 5 MB upload limit.")
    return await _predict_image(file, payload)


@app.post("/api/predict/alphabet", response_model=PredictionResponse)
async def predict_alphabet(file: Annotated[UploadFile, File(description="JPEG, PNG, or WebP hand image")]) -> PredictionResponse:
    """Use Roboflow when configured, otherwise use the bundled photo model."""
    payload = await _read_upload(file, MAX_UPLOAD_BYTES, "Image exceeds the 5 MB upload limit.")
    return await _predict_image(file, payload)


@app.post("/api/predict/roboflow", response_model=PredictionResponse)
async def predict_roboflow(file: Annotated[UploadFile, File(description="JPEG, PNG, or WebP hand image")]) -> PredictionResponse:
    """Explicit Roboflow endpoint; unlike /alphabet it never falls back locally."""
    available, detail = roboflow_predictor.availability()
    if not available:
        raise HTTPException(503, detail)
    payload = await _read_upload(file, MAX_UPLOAD_BYTES, "Image exceeds the 5 MB upload limit.")
    return await _predict_image(file, payload, force_roboflow=True)


@app.post("/api/predict/fingerspelling", response_model=PredictionResponse)
def predict_fingerspelling(payload: LandmarkRequest) -> PredictionResponse:
    """Classify 126-value MediaPipe hand-landmark vectors (wrist-relative).

    Send one vector as ``landmarks`` or a burst of them as ``frames``; with
    several frames the per-frame probability distributions are averaged
    (soft vote) before ranking, which is much more robust to detection
    jitter than a single-frame classification.
    """
    frames = payload.all_frames()
    if not frames:
        raise HTTPException(422, f"Provide 'landmarks' ({FEATURE_DIM} values) or 'frames' (a list of such vectors).")
    try:
        if len(frames) == 1:
            result = fingerspelling_predictor.predict(frames[0], top_k=5)
        else:
            result = fingerspelling_predictor.predict_frames(frames, top_k=5)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        LOGGER.exception("Fingerspelling prediction failed")
        raise HTTPException(503, str(exc)) from exc
    return _response(result, _confidence_threshold("CONFIDENCE_THRESHOLD", "0.60"))


@app.post("/api/predict/word", response_model=PredictionResponse)
async def predict_word(file: Annotated[UploadFile, File(description="Short WebM/MP4 clip of one signed word")]) -> PredictionResponse:
    if file.content_type not in VIDEO_CONTENT_TYPES:
        raise HTTPException(415, "Please upload a WebM, MP4, MOV, or AVI video clip.")
    # Check deployment state up front so a missing checkpoint or optional
    # dependency returns the same actionable message as /api/health instead of
    # whichever import happens to fail first deep inside the pipeline.
    word_available, word_detail = word_predictor.availability()
    if not word_available:
        raise HTTPException(503, f"Word model is unavailable: {word_detail}")
    payload = await _read_upload(file, MAX_VIDEO_BYTES, "Video exceeds the 32 MB upload limit.")
    if not payload:
        raise HTTPException(400, "The uploaded video is empty.")
    try:
        # The I3D forward pass takes seconds on CPU; keep the event loop free.
        result = await run_in_threadpool(word_predictor.predict_video, payload, _video_suffix(file), 5)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        LOGGER.exception("Word prediction failed")
        raise HTTPException(503, str(exc)) from exc
    return _response(result, _confidence_threshold("WORD_CONFIDENCE_THRESHOLD", "0.30"))


# --- Combined (all three models, one soft-voting prediction) ---------------


def _video_suffix(file: UploadFile) -> str:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        suffix = ".webm" if file.content_type == "video/webm" else ".mp4"
    return suffix


def _parse_landmarks_field(raw: str | None) -> tuple[list[list[float]] | None, str | None]:
    """Parse the JSON landmark form field into (frames, error).

    Accepts either one flat array of ``FEATURE_DIM`` numbers (the original
    contract) or an array of such arrays — a burst of samples of the same
    sign that the fingerspelling model will soft-vote across.
    """
    if raw is None or not raw.strip():
        return None, None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None, f"Landmarks must be a JSON array of {FEATURE_DIM} numbers."
    error = (
        f"Landmarks must be a JSON array of exactly {FEATURE_DIM} finite numbers, "
        f"or an array of up to {MAX_LANDMARK_FRAMES} such arrays."
    )
    if not isinstance(value, list) or not value:
        return None, error
    nested = isinstance(value[0], list)
    try:
        if nested:
            return LandmarkRequest(frames=value).frames, None
        return [LandmarkRequest(landmarks=value).landmarks], None
    except ValidationError:
        return None, error


def _success_source(model: str, results: list[Prediction]) -> ModelSource:
    return ModelSource(
        model,
        ran=True,
        ok=True,
        label=results[0].label,
        confidence=results[0].confidence,
        top_predictions=results,
        detail="ok",
    )


async def _run_alphabet_source(file: UploadFile) -> ModelSource:
    use_roboflow = _roboflow_available()
    if not use_roboflow and not alphabet_predictor.available:
        return ModelSource("alphabet", ran=True, detail="Alphabet model files are missing.")
    try:
        payload = await _read_upload(file, MAX_UPLOAD_BYTES, "Image exceeds the 5 MB upload limit.")
        image = _decode_image(file, payload)
        if use_roboflow:
            result = await run_in_threadpool(roboflow_predictor.predict, image, 5)
        else:
            result = await run_in_threadpool(alphabet_predictor.predict, image, len(alphabet_predictor.labels))
    except HTTPException as exc:
        return ModelSource("alphabet", ran=True, detail=str(exc.detail), client_error=True)
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.exception("Image prediction failed in combined request")
        return ModelSource("alphabet", ran=True, detail=f"Image model failed: {exc}")
    return _success_source("alphabet", result)


async def _run_fingerspelling_source(raw: str) -> ModelSource:
    frames, error = _parse_landmarks_field(raw)
    if error:
        return ModelSource("fingerspelling", ran=True, detail=error, client_error=True)
    if not fingerspelling_predictor.available:
        return ModelSource("fingerspelling", ran=True, detail="Fingerspelling model files are missing.")
    try:
        if len(frames) == 1:
            result = await run_in_threadpool(
                fingerspelling_predictor.predict, frames[0], len(fingerspelling_predictor.labels)
            )
        else:
            result = await run_in_threadpool(
                fingerspelling_predictor.predict_frames, frames, len(fingerspelling_predictor.labels)
            )
    except ValueError as exc:
        return ModelSource("fingerspelling", ran=True, detail=str(exc), client_error=True)
    except (RuntimeError, OSError) as exc:
        LOGGER.exception("Fingerspelling prediction failed in combined request")
        return ModelSource("fingerspelling", ran=True, detail=f"Fingerspelling model failed: {exc}")
    return _success_source("fingerspelling", result)


async def _run_word_source(file: UploadFile) -> ModelSource:
    word_available, word_detail = word_predictor.availability()
    if not word_available:
        return ModelSource("word", ran=True, detail=f"Word model is unavailable: {word_detail}")
    if file.content_type not in VIDEO_CONTENT_TYPES:
        return ModelSource(
            "word", ran=True, detail="Please upload a WebM, MP4, MOV, or AVI video clip.", client_error=True
        )
    try:
        payload = await _read_upload(file, MAX_VIDEO_BYTES, "Video exceeds the 32 MB upload limit.")
        if not payload:
            return ModelSource("word", ran=True, detail="The uploaded video is empty.", client_error=True)
        result = await run_in_threadpool(word_predictor.predict_video, payload, _video_suffix(file), 5)
    except HTTPException as exc:
        return ModelSource("word", ran=True, detail=str(exc.detail), client_error=True)
    except (ValueError, RuntimeError, OSError) as exc:
        LOGGER.exception("Word prediction failed in combined request")
        return ModelSource("word", ran=True, detail=f"Word model failed: {exc}")
    return _success_source("word", result)


@app.post("/api/predict/combined", response_model=CombinedPredictionResponse)
async def predict_combined(
    image: Annotated[
        UploadFile | None, File(description="JPEG/PNG/WebP frame for the alphabet photo model")
    ] = None,
    landmarks: Annotated[
        str | None, Form(description=f"JSON array of {FEATURE_DIM} wrist-relative hand-landmark values")
    ] = None,
    video: Annotated[UploadFile | None, File(description="Short WebM/MP4 clip for the word model")] = None,
) -> CombinedPredictionResponse:
    """Run every provided input through its model and combine them into one prediction.

    Send any mix of ``image`` (alphabet CNN), ``landmarks`` (fingerspelling
    MLP), and ``video`` (word model). Each model that gets input runs, the two
    letter models are merged with a soft vote, and the word model is kept as
    its own candidate; see :mod:`app.services.ensemble` for the merge policy.
    """
    sources: list[ModelSource] = []
    if image is not None and (image.filename or image.content_type):
        sources.append(await _run_alphabet_source(image))
    else:
        sources.append(not_run("alphabet"))

    if landmarks is not None and landmarks.strip():
        sources.append(await _run_fingerspelling_source(landmarks))
    else:
        sources.append(not_run("fingerspelling"))

    if video is not None and (video.filename or video.content_type):
        sources.append(await _run_word_source(video))
    else:
        sources.append(not_run("word"))

    attempted = [source for source in sources if source.ran]
    if not attempted:
        raise HTTPException(
            400, "Provide at least one input: an image ('image'), landmarks ('landmarks'), or a clip ('video')."
        )
    try:
        combined = combine(
            sources,
            _confidence_threshold("CONFIDENCE_THRESHOLD", "0.60"),
            _confidence_threshold("WORD_CONFIDENCE_THRESHOLD", "0.30"),
        )
    except EnsembleError:
        details = "; ".join(dict.fromkeys(source.detail for source in attempted if not source.ok))
        status = 400 if all(source.client_error for source in attempted) else 503
        raise HTTPException(status, f"Combined recognition failed: {details}")
    return CombinedPredictionResponse(
        label=combined.label,
        confidence=combined.confidence,
        accepted=combined.accepted,
        agreement=combined.agreement,
        method=combined.method,
        word=(
            PredictionItem(label=combined.word.label, confidence=combined.word.confidence)
            if combined.word
            else None
        ),
        sources=[
            CombinedSourceItem(
                model=source.model,
                ran=source.ran,
                ok=source.ok,
                label=source.label,
                confidence=source.confidence,
                detail=source.detail,
            )
            for source in combined.sources
        ],
        top_predictions=list(combined.top_predictions),
    )


@app.get("/api/info")
def info() -> dict[str, object]:
    return {
        "recognition": "ISL fingerspelling (photo or hand landmarks), optional Roboflow object detection, CISLR word video recognition, and a combined ensemble",
        "labels": alphabet_predictor.labels,
        "input_guidance": "Keep one hand centered in the guide with a plain, well-lit background.",
        "features": {
            "combined_recognition": "POST /api/predict/combined with any mix of image, landmarks, and video",
            "roboflow_recognition": "POST /api/predict/roboflow (enabled by ROBOFLOW_API_KEY)",
            "continuous_recognition": "POST /api/continuous/session then stream landmark frames",
            "translation_languages": list(translator.LANGUAGES.keys()),
            "emergency_phrases": "GET /api/emergency/phrases (prototype; not real dispatch)",
        },
        "providers": {
            "roboflow": {
                "configured": roboflow_predictor.configured,
                "model_id": roboflow_predictor.model_id,
                "api_url": roboflow_predictor.api_url,
            }
        },
        "models": {
            "alphabet": {
                "input": "single image",
                "labels": len(alphabet_predictor.labels),
                "guidance": "Hold one alphabet sign steady inside the guide. When configured, the Roboflow detector is used for this image request.",
            },
            "fingerspelling": {
                "input": "hand landmarks extracted in the browser",
                "labels": len(fingerspelling_predictor.labels),
                "guidance": "Keep the whole hand visible; recognition uses the hand skeleton and ignores the background.",
            },
            "word": {
                "input": "short video clip",
                "labels": len(word_predictor.labels),
                "guidance": "Record one continuous sign lasting about three seconds.",
            },
        },
    }


app.include_router(language_routes.build_router(translator))
app.dependency_overrides[continuous_routes.get_predictor] = lambda: fingerspelling_predictor
app.include_router(continuous_routes.router)
app.include_router(emergency_routes.build_router(translator))
app.dependency_overrides[emergency_routes.get_notifier] = lambda: twilio_notifier

# Mount assets after API routes so /api is never swallowed by the static app.
app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def spa(path: str = "") -> FileResponse:
    requested = (FRONTEND / path).resolve()
    if path and requested.is_relative_to(FRONTEND.resolve()) and requested.is_file():
        return FileResponse(requested)
    return FileResponse(FRONTEND / "index.html")
