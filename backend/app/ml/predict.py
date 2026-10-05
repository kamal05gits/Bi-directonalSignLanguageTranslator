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
