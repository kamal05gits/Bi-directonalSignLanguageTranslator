"""Roboflow-hosted object detection for the optional ISL model.

The Universe project at ``indiansigns/indian-sign-language_40`` exposes
model version ``indian-sign-language_40/1``.  The API key is deliberately read
only from the server environment: it must never be sent to the browser or
committed to the repository.

Roboflow returns object detections rather than one classification.  Signora
turns the detections into the same ranked ``Prediction`` shape used by the
bundled classifiers by sorting by confidence and keeping one result per class.
That lets the existing UI and message builder use this hosted model without
pretending that it recognises landmarks or video.
"""

from __future__ import annotations

import base64
import os
import threading
from typing import Any

import requests
from PIL import Image

from .keras_classifier import Prediction

DEFAULT_API_URL = "https://serverless.roboflow.com"
DEFAULT_MODEL_ID = "indian-sign-language_40/1"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_OVERLAP = 0.50
REQUEST_TIMEOUT_SECONDS = 30


class RoboflowPredictor:
    """Call a Roboflow object-detection model without exposing its API key."""

    name = "Roboflow ISL"

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str | None = None,
        api_url: str | None = None,
        confidence: float | None = None,
        overlap: float | None = None,
    ) -> None:
        self.api_key = (api_key if api_key is not None else os.getenv("ROBOFLOW_API_KEY", "")).strip()
        self.model_id = (model_id if model_id is not None else os.getenv("ROBOFLOW_MODEL_ID", DEFAULT_MODEL_ID)).strip()
        self.api_url = (api_url if api_url is not None else os.getenv("ROBOFLOW_API_URL", DEFAULT_API_URL)).rstrip("/")
        self.confidence = _float_env_or_value(
            confidence, "ROBOFLOW_CONFIDENCE", str(DEFAULT_CONFIDENCE), minimum=0.0, maximum=1.0
        )
        self.overlap = _float_env_or_value(
            overlap, "ROBOFLOW_OVERLAP", str(DEFAULT_OVERLAP), minimum=0.0, maximum=1.0
        )
        self._lock = threading.Lock()
        self._request_session: requests.Session | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model_id and self.api_url)

    @property
    def loaded(self) -> bool:
        return self._request_session is not None

    @property
    def available(self) -> bool:
        return self.availability()[0]

    def availability(self) -> tuple[bool, str]:
        """Report configuration state without making a paid/network request."""
        if not self.api_key:
            return False, "Roboflow is not configured. Set ROBOFLOW_API_KEY to enable it."
        if not self.model_id:
            return False, "ROBOFLOW_MODEL_ID is empty."
        if not self.api_url:
            return False, "ROBOFLOW_API_URL is empty."
        return True, f"ready (Roboflow model {self.model_id})"

    def _session(self) -> requests.Session:
        if self._request_session is None:
            with self._lock:
                if self._request_session is None:
                    session = requests.Session()
                    session.headers.update(
                        {
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        }
                    )
                    self._request_session = session
        return self._request_session

    def predict(self, image: Image.Image, top_k: int = 5) -> list[Prediction]:
        """Send one RGB image and return ranked, de-duplicated detections."""
        available, detail = self.availability()
        if not available:
            raise RuntimeError(detail)

        # Roboflow's serverless API accepts the base64 image as the request
        # body. Keeping the upload in memory avoids leaving camera frames on
        # the Render filesystem.
        from io import BytesIO

        encoded = BytesIO()
        image.convert("RGB").save(encoded, format="JPEG", quality=92)
        endpoint = f"{self.api_url}/{self.model_id}"
        try:
            response = self._session().post(
                endpoint,
                params={
                    "confidence": self.confidence * 100,
                    "overlap": self.overlap * 100,
                },
                data=base64.b64encode(encoded.getvalue()),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code >= 400:
                # Do not include response text: upstream errors can echo URL
                # query parameters, and this method must never leak a secret.
                raise RuntimeError(f"Roboflow request failed with HTTP {response.status_code}.")
            payload = response.json()
        except requests.RequestException as exc:
            raise RuntimeError("Could not reach the Roboflow inference service.") from exc
        except ValueError as exc:
            raise RuntimeError("Roboflow returned an invalid JSON response.") from exc

        return _predictions_from_response(payload, top_k=top_k)


def _float_env_or_value(
    value: float | None, env_name: str, default: str, *, minimum: float, maximum: float
) -> float:
    raw = str(value) if value is not None else os.getenv(env_name, default)
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        parsed = float(default)
    return min(maximum, max(minimum, parsed))


def _predictions_from_response(payload: Any, top_k: int = 5) -> list[Prediction]:
    """Convert Roboflow's detection JSON into Signora predictions."""
    if isinstance(payload, list):
        payload = payload[0] if payload and isinstance(payload[0], dict) else {}
    if not isinstance(payload, dict):
        raise RuntimeError("Roboflow returned an unexpected response shape.")

    raw_predictions = payload.get("predictions", [])
    if not isinstance(raw_predictions, list):
        raise RuntimeError("Roboflow returned an invalid predictions list.")

    ranked: list[Prediction] = []
    seen: set[str] = set()
    for item in sorted(
        (item for item in raw_predictions if isinstance(item, dict)),
        key=lambda item: float(item.get("confidence", 0.0) or 0.0),
        reverse=True,
    ):
        label = item.get("class", item.get("label"))
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if not isinstance(label, str) or not label.strip() or not 0 <= confidence <= 1:
            continue
        label = label.strip()
        if label in seen:
            continue
        seen.add(label)
        ranked.append(Prediction(label=label, confidence=confidence))
        if len(ranked) >= max(1, top_k):
            break

    if not ranked:
        raise ValueError("Roboflow found no sign in this image. Try moving your hand into the guide.")
    return ranked
