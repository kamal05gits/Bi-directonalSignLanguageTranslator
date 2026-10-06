"""CISLR word-level recognition: short video -> I3D features -> Keras classifier.

The pipeline mirrors ``backend/final_demo_inference.py``:

1. Decode only the 90 sampled frames of the uploaded video (bounded memory).
2. Resize each frame so the shorter side is 224 px, then center-crop to 224x224.
3. Extract 1024-d features with the I3D network (WLASL asl2000 checkpoint).
4. Resample the feature sequence to (32, 1024) and normalize with the
   training statistics stored in ``CISLR_NORMALIZATION.npz``.
5. Classify with ``CISLR_MODEL.keras`` into one of the CISLR word labels.

PyTorch, OpenCV, and the 57 MB I3D checkpoint are heavyweight optional
dependencies. Everything is imported lazily, and missing pieces are reported
honestly through :meth:`WordPredictor.availability` instead of failing at
import time, so the rest of the service keeps working without them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from .keras_classifier import LazyKerasClassifier, Prediction

LOGGER = logging.getLogger(__name__)

NUM_INPUT_FRAMES = 90
TARGET_FEATURE_FRAMES = 32
IMAGE_SIZE = 224
I3D_FEATURE_DIM = 1024
I3D_NUM_CLASSES = 2000
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"
DEPENDENCY_HINT = "Install the optional dependencies with: pip install -r backend/requirements-word.txt"
DEFAULT_CHECKPOINT_SHA256 = "243a19e6deef3becffbfc5b7dd8adb32916c8bee482565ca243c082584732620"
MAX_CHECKPOINT_BYTES = 100 * 1024 * 1024


def is_lfs_pointer(path: Path) -> bool:
    """Return True when ``path`` holds a Git LFS pointer instead of the real file."""
    try:
        with Path(path).open("rb") as handle:
            return handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX
    except OSError:
        return False


class WordPredictor(LazyKerasClassifier):
    """Recognize a signed word from a short video clip."""

    name = "word"

    def __init__(
        self,
        model_path,
        labels_path,
        normalization_path,
        weights_path,
        i3d_code_dir,
        checkpoint_url: str | None = None,
        checkpoint_sha256: str = DEFAULT_CHECKPOINT_SHA256,
    ) -> None:
        super().__init__(model_path, labels_path)
        self.normalization_path = Path(normalization_path)
        self.weights_path = Path(weights_path)
        self.i3d_code_dir = Path(i3d_code_dir)
        self.checkpoint_url = checkpoint_url
        self.checkpoint_sha256 = checkpoint_sha256.lower()
        self._i3d: Any = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._i3d_lock = threading.Lock()

    # ------------------------------------------------------------------ status

    def availability(self) -> tuple[bool, str]:
        """Report whether a prediction can run, with an actionable reason."""
        if not self.model_path.is_file():
            return False, "Word classifier (backend/models/cislr) is not deployed in this build."
        if not self.labels:
            return False, "Word labels (CISLR_LABELS.json) are missing."
        if not self.normalization_path.is_file():
            return False, "Word normalization stats (CISLR_NORMALIZATION.npz) are missing."
        checkpoint_missing = not self.weights_path.is_file() or is_lfs_pointer(self.weights_path)
        if checkpoint_missing and not self.checkpoint_url:
            return False, "I3D checkpoint is missing or still a Git LFS pointer. Restore it with: git lfs pull"
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("cv2") is None:
            return False, f"PyTorch/OpenCV are not installed. {DEPENDENCY_HINT}"
        if checkpoint_missing:
            return True, "ready (the I3D checkpoint will be downloaded securely on the first prediction)"
        return True, "ready"

    def _ensure_checkpoint(self) -> None:
        """Materialize and verify the LFS checkpoint when configured to do so.

        Source checkouts commonly contain only Git LFS's 133-byte pointer,
        which previously made the Word tab permanently unavailable even when
        all Python dependencies were installed.  The production image still
        downloads at build time, while local/source deployments can now fetch
        the same file lazily.  A size cap, pinned SHA-256, temporary file and
        atomic replace prevent partial or unexpected files being loaded.
        """
        if self.weights_path.is_file() and not is_lfs_pointer(self.weights_path):
            return
        if not self.checkpoint_url:
            raise RuntimeError("I3D checkpoint is missing. Restore it with: git lfs pull")
        with self._i3d_lock:
            if self.weights_path.is_file() and not is_lfs_pointer(self.weights_path):
                return
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path: Path | None = None
            try:
                request = urllib.request.Request(
                    self.checkpoint_url,
                    headers={"User-Agent": "Signora/2.2 checkpoint downloader"},
                )
                with urllib.request.urlopen(request, timeout=120) as response, tempfile.NamedTemporaryFile(
                    dir=self.weights_path.parent, suffix=".download", delete=False
                ) as output:
                    temp_path = Path(output.name)
                    digest = hashlib.sha256()
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_CHECKPOINT_BYTES:
                            raise RuntimeError("I3D checkpoint download exceeded the 100 MB safety limit.")
                        digest.update(chunk)
                        output.write(chunk)
                if total == 0:
                    raise RuntimeError("I3D checkpoint download was empty.")
                actual = digest.hexdigest()
                if self.checkpoint_sha256 and actual != self.checkpoint_sha256:
                    raise RuntimeError(
                        "I3D checkpoint checksum mismatch; the downloaded file was discarded "
                        f"(expected {self.checkpoint_sha256}, got {actual})."
                    )
                os.replace(temp_path, self.weights_path)
                temp_path = None
                LOGGER.info("Downloaded and verified I3D checkpoint (%d bytes)", total)
            except (OSError, urllib.error.URLError) as exc:
                raise RuntimeError(f"Could not download the I3D checkpoint: {exc}") from exc
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

    # --------------------------------------------------------------- lazy loads

    def _load_normalization(self) -> tuple[np.ndarray, np.ndarray]:
        if self._mean is not None and self._std is not None:
            return self._mean, self._std
        with self._i3d_lock:
            if self._mean is None or self._std is None:
                if not self.normalization_path.is_file():
                    raise RuntimeError(f"Word normalization file is missing: {self.normalization_path}")
                archive = np.load(self.normalization_path)
                if "mean" not in archive or "std" not in archive:
                    raise RuntimeError("Word normalization file must contain 'mean' and 'std' arrays.")
                self._mean = archive["mean"].astype(np.float32).reshape(1, I3D_FEATURE_DIM)
                std = archive["std"].astype(np.float32).reshape(1, I3D_FEATURE_DIM)
                self._std = np.where(std < 1e-6, 1.0, std)
        return self._mean, self._std

    def _load_i3d(self):
        if self._i3d is not None:
            return self._i3d
        with self._i3d_lock:
            if self._i3d is None:
                try:
                    import torch
                except ImportError as exc:
                    raise RuntimeError(f"PyTorch is required for the word model. {DEPENDENCY_HINT}") from exc
                if not self.weights_path.is_file():
                    raise RuntimeError(f"I3D checkpoint is missing: {self.weights_path}. Restore it with: git lfs pull")
                if is_lfs_pointer(self.weights_path):
                    raise RuntimeError("I3D checkpoint is a Git LFS pointer. Download the real file with: git lfs pull")
                code_dir = str(self.i3d_code_dir)
                if code_dir not in sys.path:
                    sys.path.insert(0, code_dir)
                try:
                    from pytorch_i3d import InceptionI3d
                except ImportError as exc:
                    raise RuntimeError(f"Could not import the I3D module from {self.i3d_code_dir}: {exc}") from exc

                checkpoint = torch.load(self.weights_path, map_location="cpu")
                if isinstance(checkpoint, dict):
                    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
                else:
                    state_dict = checkpoint
                cleaned = {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}

                i3d = InceptionI3d(num_classes=I3D_NUM_CLASSES, in_channels=3)
                missing, unexpected = i3d.load_state_dict(cleaned, strict=False)
                LOGGER.info("I3D checkpoint loaded (missing keys: %d, unexpected keys: %d)", len(missing), len(unexpected))
                i3d.eval()
                self._i3d = i3d
        return self._i3d

    # ---------------------------------------------------------------- pipeline

    @staticmethod
    def _import_cv2():
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(f"OpenCV is required for the word model. {DEPENDENCY_HINT}") from exc
        return cv2

    @classmethod
    def _open_capture(cls, cv2, video_path: Path):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("Could not decode the uploaded video. Try WebM or MP4.")
        return capture

    @classmethod
    def _count_frames(cls, cv2, video_path: Path) -> int:
        """Count decodable frames without keeping any of them in memory.

        ``grab()`` advances the decoder without copying/converting the frame,
        and the container metadata (``CAP_PROP_FRAME_COUNT``) is unreliable for
        browser-recorded WebM, so it is only used as a sanity hint.
        """
        capture = cls._open_capture(cv2, video_path)
        try:
            total = 0
            while capture.grab():
                total += 1
        finally:
            capture.release()
        return total

    @classmethod
    def _sample_frames(cls, video_path: Path) -> np.ndarray:
        """Return ``NUM_INPUT_FRAMES`` preprocessed frames as (T, 224, 224, 3).

        Only the sampled frames are ever decoded into memory (~54 MB at
        224x224 float32), instead of holding the whole clip as full-resolution
        RGB arrays, which can reach several GB for a long 1080p upload.
        The sampling positions are identical to decoding everything first and
        indexing with ``np.linspace(0, total - 1, NUM_INPUT_FRAMES)``.
        """
        cv2 = cls._import_cv2()
        total = cls._count_frames(cv2, video_path)
        if total == 0:
            raise ValueError("The uploaded video contains no readable frames.")

        indices = np.linspace(0, total - 1, NUM_INPUT_FRAMES).astype(np.int32)
        wanted = sorted(set(int(index) for index in indices))

        decoded: dict[int, np.ndarray] = {}
        capture = cls._open_capture(cv2, video_path)
        try:
            position = 0
            for index in wanted:
                while position < index and capture.grab():
                    position += 1
                ok, frame = capture.read()
                if not ok:
                    break  # Second pass ended early; fall back to what we have.
                position += 1
                decoded[index] = cls._preprocess_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()

        if not decoded:
            raise ValueError("The uploaded video contains no readable frames.")

        available = sorted(decoded)
        sampled = [
            decoded[int(index)] if int(index) in decoded
            else decoded[min(available, key=lambda kept: abs(kept - int(index)))]
            for index in indices
        ]
        return np.stack(sampled)

    @classmethod
    def _preprocess_frame(cls, frame: np.ndarray) -> np.ndarray:
        cv2 = cls._import_cv2()
        height, width = frame.shape[:2]
        scale = IMAGE_SIZE / min(height, width)
        new_w, new_h = int(round(width * scale)), int(round(height * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        y0, x0 = max((new_h - IMAGE_SIZE) // 2, 0), max((new_w - IMAGE_SIZE) // 2, 0)
        cropped = resized[y0:y0 + IMAGE_SIZE, x0:x0 + IMAGE_SIZE]
        cropped = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        return cropped.astype(np.float32) / 127.5 - 1.0

    def _extract_features(self, sampled: np.ndarray) -> np.ndarray:
        """Run the I3D feature extractor over (NUM_INPUT_FRAMES, 224, 224, 3) input."""
        import torch

        if sampled.shape[0] != NUM_INPUT_FRAMES:
            raise RuntimeError(f"Expected {NUM_INPUT_FRAMES} preprocessed frames, got {sampled.shape[0]}.")
        # T,H,W,C -> B,C,T,H,W
        tensor = torch.from_numpy(np.transpose(sampled, (3, 0, 1, 2))).unsqueeze(0).float()

        i3d = self._load_i3d()
        with self._i3d_lock, torch.no_grad():
            output = i3d.extract_features(tensor)
        if output.ndim == 5:  # B,C,T,H,W -> B,C,T
            output = output.mean(dim=(-1, -2))
        if output.ndim != 3:
            raise RuntimeError(f"Unexpected I3D output shape: {tuple(output.shape)}")
        features = output[0].cpu().numpy().T  # (T, 1024)
        if features.shape[1] != I3D_FEATURE_DIM:
            raise RuntimeError(f"Expected {I3D_FEATURE_DIM} I3D features, got {features.shape[1]}.")

        source_positions = np.linspace(0, features.shape[0] - 1, features.shape[0])
        target_positions = np.linspace(0, features.shape[0] - 1, TARGET_FEATURE_FRAMES)
        resampled = np.stack(
            [np.interp(target_positions, source_positions, features[:, d]) for d in range(I3D_FEATURE_DIM)],
            axis=1,
        )
        return resampled.astype(np.float32)  # (32, 1024)

    def predict_video(self, payload: bytes, suffix: str, top_k: int = 5) -> list[Prediction]:
        """Classify an uploaded video clip into ranked CISLR word labels."""
        # Do this before decoding the user's clip so a deployment with an LFS
        # pointer becomes usable automatically and fails without wasted work.
        self._ensure_checkpoint()
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(payload)
                temp_path = Path(handle.name)
            features = self._extract_features(self._sample_frames(temp_path))
            mean, std = self._load_normalization()
            normalized = np.nan_to_num((features - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
            return self._predict_array(normalized[np.newaxis, ...].astype(np.float32), top_k)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
