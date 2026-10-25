"""Temporal filtering and commit gating for live gesture predictions."""
from collections import Counter, deque
from math import isfinite
from typing import Deque, Dict, Optional, Tuple


class PredictionSmoother:
    """Return a label only after enough recent, confident predictions agree."""

    def __init__(self, window_size: int, min_stable_frames: int, confidence_threshold: float):
        if window_size < 1:
            raise ValueError('window_size must be at least 1.')
        if not 1 <= min_stable_frames <= window_size:
            raise ValueError('min_stable_frames must be between 1 and window_size.')
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError('confidence_threshold must be between 0 and 1.')

        self.window_size = window_size
        self.min_stable_frames = min_stable_frames
        self.confidence_threshold = confidence_threshold
        self._predictions: Deque[Tuple[str, float, Dict]] = deque(maxlen=window_size)

    def reset(self) -> None:
        self._predictions.clear()

    def update(self, prediction: Optional[Dict]) -> Optional[Dict]:
        if not prediction or prediction.get('hand_count', 0) <= 0:
            self.reset()
            return None

        label = prediction.get('gesture')
        confidence = prediction.get('confidence')
        if (
            not isinstance(label, str)
            or not label
            or not isinstance(confidence, (int, float))
            or not isfinite(confidence)
            or confidence <= self.confidence_threshold
        ):
            self.reset()
            return None

        self._predictions.append((label, float(confidence), prediction))
        counts = Counter(item[0] for item in self._predictions)
        newest_first = {}
        for index, item in enumerate(self._predictions):
            newest_first[item[0]] = index
        stable_label = max(counts, key=lambda item: (counts[item], newest_first[item]))
        stable_count = counts[stable_label]
        if stable_count < self.min_stable_frames:
            return None

        matching = [item for item in self._predictions if item[0] == stable_label]
        smoothed = dict(matching[-1][2])
        smoothed['gesture'] = stable_label
        smoothed['confidence'] = sum(item[1] for item in matching) / len(matching)
        smoothed['stable_frames'] = stable_count
        return smoothed


class StablePredictionCommitter:
    """Prevent repeat-frame commits while allowing a held sign to be released."""

    def __init__(self, cooldown_seconds: float, same_gesture_cooldown_seconds: float):
        if cooldown_seconds < 0 or same_gesture_cooldown_seconds < 0:
            raise ValueError('Prediction cooldowns cannot be negative.')
        self.cooldown_seconds = cooldown_seconds
        self.same_gesture_cooldown_seconds = same_gesture_cooldown_seconds
        self._last_label: Optional[str] = None
        self._last_commit_time: Optional[float] = None
        self._blocked_label: Optional[str] = None

    def reset(self) -> None:
        self._last_label = None
        self._last_commit_time = None
        self._blocked_label = None

    def release(self) -> None:
        self._blocked_label = None

    def should_commit(self, label: str, now: float) -> bool:
        if not label or label == self._blocked_label:
            return False

        cooldown = (
            self.same_gesture_cooldown_seconds
            if label == self._last_label
            else self.cooldown_seconds
        )
        if self._last_commit_time is not None and now - self._last_commit_time < cooldown:
            return False

        self._last_label = label
        self._last_commit_time = now
        self._blocked_label = label
        return True
