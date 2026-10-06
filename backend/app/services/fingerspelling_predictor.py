"""Landmark-vector inference for the fingerspelling MLP classifier."""

from __future__ import annotations

import numpy as np

from .keras_classifier import LazyKerasClassifier, Prediction

# 2 hands x 21 landmarks x (x, y, z); matches app.vision.landmarks.FEATURE_DIM.
FEATURE_DIM = 126
HAND_DIM = FEATURE_DIM // 2


def mirror_hands(vector: np.ndarray) -> np.ndarray:
    """Swap the left/right hand slots and mirror the x axis.

    This maps a sign made with one physical hand onto the representation the
    *other* hand would have produced. MediaPipe's handedness label flips
    depending on whether the camera feed is mirrored, and users may simply
    sign with the opposite hand from the training data — either way the pose
    lands in the hand slot the model never saw, producing a confidently
    wrong prediction. Evaluating both chiralities and keeping the stronger
    one rescues those cases.
    """
    mirrored = np.empty_like(vector)
    mirrored[:HAND_DIM] = vector[HAND_DIM:]
    mirrored[HAND_DIM:] = vector[:HAND_DIM]
    mirrored[0::3] = -mirrored[0::3]  # negate every x coordinate
    return mirrored


class FingerspellingPredictor(LazyKerasClassifier):
    """Classify a 126-value hand-landmark vector into an ISL alphabet letter.

    The vector layout matches ``app.vision.landmarks.extract_landmark_features``:
    the left hand occupies values 0-62 and the right hand values 63-125, with
    each point stored as wrist-relative ``(x, y, z)`` and invisible hands left
    as zeros. Landmarks are extracted in the browser with MediaPipe, so the
    server needs neither a camera nor MediaPipe itself.

    ``temperature`` softens the model's (heavily saturated) softmax so
    runner-up signs keep honest probabilities, and ``mirror_tta`` evaluates
    the hand-swapped mirror of each input as well, adopting it only when it
    clearly outscores the original (``mirror_margin``).
    """

    name = "fingerspelling"

    def __init__(
        self,
        model_path,
        labels_path,
        feature_dim: int = FEATURE_DIM,
        temperature: float = 1.0,
        mirror_tta: bool = False,
        mirror_margin: float = 1.25,
    ) -> None:
        super().__init__(model_path, labels_path, temperature=temperature)
        self.feature_dim = feature_dim
        self.mirror_tta = mirror_tta
        self.mirror_margin = mirror_margin

    def _validated(self, landmarks) -> np.ndarray:
        vector = np.asarray(landmarks, dtype=np.float32).reshape(-1)
        if vector.shape != (self.feature_dim,):
            raise ValueError(
                f"Expected {self.feature_dim} landmark values "
                f"(2 hands x 21 points x 3 coordinates), got {vector.size}."
            )
        if not np.isfinite(vector).all():
            raise ValueError("Landmark values must be finite numbers.")
        return vector

    def _consensus(self, vectors: list[np.ndarray]) -> np.ndarray:
        """Average calibrated distributions, optionally soft-voting chirality.

        With ``mirror_tta`` enabled the hand-swapped mirror of every frame is
        classified in the same batch; the mirrored consensus replaces the
        original one only when its top score is at least ``mirror_margin``
        times higher, so correctly-slotted signs are never destabilized while
        wrong-slot/opposite-hand signs stop being confidently misread.
        """
        batch = list(vectors)
        if self.mirror_tta:
            batch += [mirror_hands(vector) for vector in vectors]
        probabilities = self._predict_probabilities(np.stack(batch))
        original = probabilities[: len(vectors)].mean(axis=0)
        if not self.mirror_tta:
            return original
        mirrored = probabilities[len(vectors):].mean(axis=0)
        return mirrored if mirrored.max() >= original.max() * self.mirror_margin else original

    def predict(self, landmarks, top_k: int = 3) -> list[Prediction]:
        vector = self._validated(landmarks)
        if not vector.any():
            raise ValueError("No hand landmarks were detected. Show one hand to the camera and try again.")
        return self._rank(self._consensus([vector]), top_k)

    def predict_frames(self, frames, top_k: int = 3) -> list[Prediction]:
        """Classify several landmark frames of the *same* sign and soft-vote.

        Every frame is validated like :meth:`predict`; all-zero frames (no
        hand visible in that sample) are dropped. The per-frame probability
        distributions are calibrated and averaged before ranking, so one
        jittery MediaPipe detection cannot flip the top prediction the way
        it can with a single-frame classification.
        """
        vectors = [self._validated(frame) for frame in frames]
        vectors = [vector for vector in vectors if vector.any()]
        if not vectors:
            raise ValueError("No hand landmarks were detected. Show one hand to the camera and try again.")
        return self._rank(self._consensus(vectors), top_k)
