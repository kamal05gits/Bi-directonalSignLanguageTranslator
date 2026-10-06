import numpy as np


class FeatureScaler:
    def __init__(self): self.mean = self.std = None
    def fit(self, values):
        flat = np.asarray(values, dtype=np.float32).reshape(-1, values.shape[-1])
        self.mean = flat.mean(axis=0); self.std = flat.std(axis=0)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        return self
    def transform(self, values):
        if self.mean is None: raise RuntimeError("Scaler has not been fitted")
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std
