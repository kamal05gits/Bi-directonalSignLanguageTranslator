from dataclasses import dataclass
import time
import numpy as np

@dataclass
class PredictionResult:
    label: str
    confidence: float
    accepted: bool
    reason: str = ""

class PredictionEngine:
    def __init__(self, model, labels, confidence_threshold=.75, cooldown_seconds=1):
        self.model, self.labels = model, labels
        self.threshold, self.cooldown = confidence_threshold, cooldown_seconds
        self._last_label, self._last_time = None, 0.0
    def predict(self, sequence):
        scores = np.asarray(self.model.predict(np.asarray(sequence)[None, ...], verbose=0))[0]
        index = int(np.argmax(scores)); label = self.labels[index]; confidence = float(scores[index])
        now = time.monotonic(); accepted = confidence >= self.threshold; reason = ""
        if not accepted: reason = "Low confidence"
        elif label == self._last_label and now - self._last_time < self.cooldown:
            accepted = False; reason = "Duplicate prediction suppressed"
        else: self._last_label, self._last_time = label, now
        return PredictionResult(label, confidence, accepted, reason)


@dataclass
class StabilizedPrediction:
    label: str | None
    confidence: float
    accepted: bool
    reason: str = ""


class LabelStabilizer:
    """Turns a noisy per-frame ``(label, confidence)`` stream into stable tokens.

    This is the "confidence-aware prediction" logic for a *live* camera feed
    (as opposed to the one-shot capture flow): a label must win
    ``hold_frames`` consecutive frames above ``confidence_threshold`` before
    it is accepted (temporal smoothing), and the same label is then
    suppressed for ``cooldown_seconds`` so holding a sign in front of the
    camera does not spam the same letter/word into the sentence (duplicate
    prevention). A different label is always accepted immediately once it
    wins its own hold streak.
    """

    def __init__(self, confidence_threshold: float = 0.70, hold_frames: int = 3, cooldown_seconds: float = 1.2) -> None:
        self.confidence_threshold = confidence_threshold
        self.hold_frames = max(1, hold_frames)
        self.cooldown_seconds = cooldown_seconds
        self._streak_label: str | None = None
        self._streak_count = 0
        self._last_accepted_label: str | None = None
        self._last_accepted_time = float("-inf")

    def update(self, label: str, confidence: float, now: float | None = None) -> StabilizedPrediction:
        now = time.monotonic() if now is None else now
        if confidence < self.confidence_threshold:
            self._streak_label, self._streak_count = None, 0
            return StabilizedPrediction(label, confidence, False, "Low confidence")

        if label == self._streak_label:
            self._streak_count += 1
        else:
            self._streak_label, self._streak_count = label, 1

        if self._streak_count < self.hold_frames:
            return StabilizedPrediction(label, confidence, False, "Stabilizing")

        if label == self._last_accepted_label and now - self._last_accepted_time < self.cooldown_seconds:
            return StabilizedPrediction(label, confidence, False, "Duplicate prediction suppressed")

        self._last_accepted_label, self._last_accepted_time = label, now
        self._streak_count = 0
        return StabilizedPrediction(label, confidence, True)

    def reset(self) -> None:
        self._streak_label, self._streak_count = None, 0
        self._last_accepted_label, self._last_accepted_time = None, float("-inf")
