from collections import deque

import numpy as np


class SequenceBuffer:
    def __init__(self, length: int, feature_dim: int):
        self.length, self.feature_dim = length, feature_dim
        self._values = deque(maxlen=length)
    def add(self, values):
        item = np.asarray(values, dtype=np.float32)
        if item.shape != (self.feature_dim,):
            raise ValueError(f"Expected shape ({self.feature_dim},), got {item.shape}")
        self._values.append(item)
    @property
    def ready(self): return len(self._values) == self.length
    def as_array(self): return np.asarray(self._values, dtype=np.float32)
    def clear(self): self._values.clear()
