import time
from dataclasses import dataclass

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


class DistributionSmoother:
    """Exponential moving average over full ranked prediction distributions.

    The per-frame classifier only has to be wrong for a single frame for the
    correct label to drop into the low-confidence suggestions. Smoothing the
    *whole* probability distribution over time (instead of only looking at
    each frame's argmax) lets a label that is consistently second place at,
    say, 40% overtake a label that spiked once at 60% — so the stabilized
    top-1 tracks what the user is actually holding, not one noisy frame.

    ``update`` takes ``(label, confidence)`` pairs (any iterable) and returns
    the smoothed ranking as a list of ``(label, confidence)`` tuples, best
    first. The first update after a ``reset`` adopts the incoming
    distribution as-is so a fresh sign is never diluted by a cold start.
    """

    def __init__(self, alpha: float = 0.45) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        self.alpha = alpha
        self._scores: dict[str, float] = {}

    def update(self, ranked) -> list[tuple[str, float]]:
        current = {str(label): float(confidence) for label, confidence in ranked}
        if not self._scores:
            self._scores = dict(current)
        else:
            labels = set(self._scores) | set(current)
            self._scores = {
                label: (1.0 - self.alpha) * self._scores.get(label, 0.0) + self.alpha * current.get(label, 0.0)
                for label in labels
            }
            # Drop labels whose evidence has decayed to noise so the state
            # cannot grow without bound during a long session.
            self._scores = {label: score for label, score in self._scores.items() if score >= 1e-4}
        return sorted(self._scores.items(), key=lambda item: item[1], reverse=True)

    def reset(self) -> None:
        self._scores = {}


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
