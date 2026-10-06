"""Continuous (live-stream) sign recognition.

This wires three previously-standalone, unit-tested-only components into a
real, callable pipeline:

- :class:`app.features.sequence_buffer.SequenceBuffer` - a rolling window of
  raw landmark frames, averaged before classification to damp per-frame
  jitter (the "rolling frame buffer" / temporal feature smoothing).
- :class:`app.ml.predict.LabelStabilizer` - turns the noisy per-frame label
  stream into stable, de-duplicated letters ("stable prediction detection" +
  "duplicate prevention").
- :class:`app.language.sentence_processor.SentenceProcessor` - accumulates
  accepted letters into editable text ("token sequencing", manual
  correction, space/punctuation insertion).

The browser posts one landmark vector per animation frame to
``/api/continuous/session/{id}/frame`` while a hand is visible; a session
holds the rolling buffer, the stabilizer, and the sentence builder so the
client stays a thin, stateless capture loop.

The fingerspelling predictor is injected through a FastAPI dependency
(``get_predictor``) rather than captured at router-construction time, so
``app.main`` wires the real bundled model while tests can override it with a
lightweight stub (see ``app.dependency_overrides`` in ``tests/test_api.py``
style) without needing TensorFlow installed.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..features.sequence_buffer import SequenceBuffer
from ..language.sentence_processor import SentenceProcessor
from ..ml.predict import DistributionSmoother, LabelStabilizer
from ..services.fingerspelling_predictor import FEATURE_DIM, FingerspellingPredictor

SESSION_TTL_SECONDS = 30 * 60
BUFFER_FRAMES = 4
SUGGESTION_COUNT = 4  # ranked alternatives returned with every frame

router = APIRouter(prefix="/api/continuous", tags=["continuous"])


@dataclass
class ContinuousSession:
    # 0.55 instead of the stabilizer's 0.70 default: fingerspelling output is
    # temperature-calibrated now, so honest confidences run lower than the
    # old saturated ones while still requiring a 3-frame hold to accept.
    stabilizer: LabelStabilizer = field(default_factory=lambda: LabelStabilizer(confidence_threshold=0.55))
    sentence: SentenceProcessor = field(default_factory=lambda: SentenceProcessor(token_mode="character"))
    buffer: SequenceBuffer = field(default_factory=lambda: SequenceBuffer(BUFFER_FRAMES, FEATURE_DIM))
    smoother: DistributionSmoother = field(default_factory=DistributionSmoother)
    last_seen: float = field(default_factory=time.monotonic)

    def reset_recognition(self) -> None:
        """Forget all smoothing state so the next sign starts fresh."""
        self.stabilizer.reset()
        self.buffer.clear()
        self.smoother.reset()


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, ContinuousSession] = {}
        self._lock = threading.Lock()

    def create(self) -> str:
        session_id = uuid.uuid4().hex
        with self._lock:
            self._gc()
            self._sessions[session_id] = ContinuousSession()
        return session_id

    def get(self, session_id: str) -> ContinuousSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            session.last_seen = time.monotonic()
            return session

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _gc(self) -> None:
        cutoff = time.monotonic() - SESSION_TTL_SECONDS
        expired = [sid for sid, session in self._sessions.items() if session.last_seen < cutoff]
        for sid in expired:
            del self._sessions[sid]


_store = SessionStore()


def get_predictor() -> FingerspellingPredictor:  # pragma: no cover - overridden in app.main
    raise RuntimeError("No fingerspelling predictor configured for continuous recognition.")


class FrameRequest(BaseModel):
    landmarks: list[float] = Field(min_length=FEATURE_DIM, max_length=FEATURE_DIM)


class PunctuationRequest(BaseModel):
    mark: str = Field(min_length=1, max_length=1)


class AppendRequest(BaseModel):
    token: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z]+$")


class SuggestionItem(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class FrameResponse(BaseModel):
    label: str | None = None
    confidence: float = 0.0
    accepted: bool = False
    reason: str = ""
    top_predictions: list[SuggestionItem] = []
    text: str
    tokens: list[str]


class StateResponse(BaseModel):
    text: str
    tokens: list[str]


def _get(session_id: str) -> ContinuousSession:
    try:
        return _store.get(session_id)
    except KeyError as exc:
        raise HTTPException(404, "Unknown session. Start one with POST /api/continuous/session.") from exc


def _state(session: ContinuousSession) -> dict:
    return {"text": session.sentence.text, "tokens": session.sentence.original_tokens}


@router.post("/session")
def create_session() -> dict:
    return {"session_id": _store.create()}


@router.get("/session/{session_id}", response_model=StateResponse)
def get_session(session_id: str) -> StateResponse:
    return StateResponse(**_state(_get(session_id)))


@router.post("/session/{session_id}/frame", response_model=FrameResponse)
def push_frame(
    session_id: str,
    payload: FrameRequest,
    predictor: FingerspellingPredictor = Depends(get_predictor),
) -> FrameResponse:
    if not predictor.available:
        raise HTTPException(503, "Fingerspelling model is unavailable, so continuous recognition cannot run.")
    session = _get(session_id)
    vector = payload.landmarks

    if not any(vector):
        # No hand in frame: reset smoothing state so a fresh sign has to
        # re-earn its hold streak instead of inheriting stale history.
        session.reset_recognition()
        return FrameResponse(label=None, confidence=0.0, accepted=False, reason="No hand detected", **_state(session))

    session.buffer.add(vector)
    if not session.buffer.ready:
        return FrameResponse(label=None, confidence=0.0, accepted=False, reason="Stabilizing", **_state(session))

    smoothed = session.buffer.as_array().mean(axis=0)
    try:
        ranked = predictor.predict(smoothed, top_k=SUGGESTION_COUNT + 1)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

    # Smooth the whole ranked distribution over time, not just each frame's
    # argmax: a label that is consistently strong overtakes one that spiked
    # on a single noisy frame, so the right sign stops hiding in the
    # low-confidence suggestions.
    consensus = session.smoother.update((item.label, item.confidence) for item in ranked)
    top_label, top_confidence = consensus[0]
    stabilized = session.stabilizer.update(top_label, top_confidence)
    if stabilized.accepted:
        session.sentence.append_token(stabilized.label)
    return FrameResponse(
        label=stabilized.label,
        confidence=stabilized.confidence,
        accepted=stabilized.accepted,
        reason=stabilized.reason,
        top_predictions=[
            SuggestionItem(label=label, confidence=min(1.0, max(0.0, confidence)))
            for label, confidence in consensus[: SUGGESTION_COUNT + 1]
        ],
        **_state(session),
    )


@router.post("/session/{session_id}/space", response_model=StateResponse)
def insert_space(session_id: str) -> StateResponse:
    session = _get(session_id)
    session.sentence.insert_space()
    session.reset_recognition()
    return StateResponse(**_state(session))


@router.post("/session/{session_id}/append", response_model=StateResponse)
def append_token(session_id: str, payload: AppendRequest) -> StateResponse:
    """Manually accept a suggested label the recognizer ranked below top-1.

    When the correct sign shows up in ``top_predictions`` with a lower
    confidence than the (wrong) leader, the user can tap it instead of
    re-signing; the smoothing state is reset so the correction does not keep
    fighting the stale consensus.
    """
    session = _get(session_id)
    session.sentence.append_token(payload.token.lower())
    session.reset_recognition()
    return StateResponse(**_state(session))


@router.post("/session/{session_id}/punctuation", response_model=StateResponse)
def insert_punctuation(session_id: str, payload: PunctuationRequest) -> StateResponse:
    session = _get(session_id)
    session.sentence.append_punctuation(payload.mark)
    return StateResponse(**_state(session))


@router.post("/session/{session_id}/backspace", response_model=StateResponse)
def backspace(session_id: str) -> StateResponse:
    session = _get(session_id)
    session.sentence.backspace()
    return StateResponse(**_state(session))


@router.post("/session/{session_id}/clear", response_model=StateResponse)
def clear(session_id: str) -> StateResponse:
    session = _get(session_id)
    session.sentence.clear()
    session.reset_recognition()
    return StateResponse(**_state(session))


@router.delete("/session/{session_id}")
def delete_session(session_id: str) -> dict:
    _store.drop(session_id)
    return {"ok": True}
