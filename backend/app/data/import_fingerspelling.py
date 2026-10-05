import numpy as np

def normalize_letter_label(label: str) -> str:
    if label == " ": return "space"
    if label == ".": return "period"
    return label.strip().lower()

def resample_sequence(frames, target_length: int) -> np.ndarray:
    values = np.asarray(frames, dtype=np.float32)
    if not len(values): raise ValueError("Cannot resample an empty sequence")
    old = np.linspace(0, 1, len(values)); new = np.linspace(0, 1, target_length)
    return np.stack([np.interp(new, old, values[:, i]) for i in range(values.shape[1])], axis=1).astype(np.float32)
