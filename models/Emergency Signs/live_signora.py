
from pathlib import Path
from collections import deque
import json
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "isl_emergency_lstm.keras"
LABEL_PATH = BASE / "class_labels.json"
TASK_PATH = BASE / "hand_landmarker.task"

SEQUENCE_LENGTH = 30
SAMPLE_EVERY = 2
PREDICT_EVERY = 5
CONFIDENCE_THRESHOLD = 0.65

with open(LABEL_PATH, "r") as f:
    labels = json.load(f)

model = tf.keras.models.load_model(MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(TASK_PATH)
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)

sequence = deque(maxlen=SEQUENCE_LENGTH)
recent_predictions = deque(maxlen=5)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permissions.")

frame_number = 0
prediction_counter = 0
display_label = "Collecting frames..."
display_confidence = 0.0

try:
    with vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            display_frame = cv2.flip(frame, 1)
            frame_number += 1

            if frame_number % SAMPLE_EVERY == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )
                result = detector.detect(image)

                features = np.zeros((2, 21, 3), dtype=np.float32)

                if result.hand_landmarks:
                    for hand_idx, landmarks in enumerate(
                        result.hand_landmarks[:2]
                    ):
                        features[hand_idx] = np.asarray(
                            [[lm.x, lm.y, lm.z] for lm in landmarks],
                            dtype=np.float32
                        )

                sequence.append(features.reshape(-1))

                if len(sequence) == SEQUENCE_LENGTH:
                    prediction_counter += 1

                    if prediction_counter % PREDICT_EVERY == 0:
                        x = np.asarray(sequence, dtype=np.float32)
                        probs = model.predict(x[None, ...], verbose=0)[0]

                        best_idx = int(np.argmax(probs))
                        confidence = float(probs[best_idx])

                        recent_predictions.append(
                            (best_idx, confidence)
                        )

                        # Smooth predictions by majority vote.
                        counts = {}
                        for idx, conf in recent_predictions:
                            counts[idx] = counts.get(idx, 0) + 1

                        chosen = max(counts, key=counts.get)
                        chosen_conf = float(np.mean([
                            conf for idx, conf in recent_predictions
                            if idx == chosen
                        ]))

                        if (
                            chosen_conf >= CONFIDENCE_THRESHOLD
                            and counts[chosen] >= 3
                        ):
                            display_label = labels[chosen]
                            display_confidence = chosen_conf
                        else:
                            display_label = "Uncertain / try again"
                            display_confidence = confidence

            cv2.putText(
                display_frame,
                f"Sign: {display_label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
            cv2.putText(
                display_frame,
                f"Confidence: {display_confidence:.2f}",
                (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
            cv2.putText(
                display_frame,
                f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
                (20, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )
            cv2.putText(
                display_frame,
                "Press Q to quit",
                (20, display_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow("SignOra - Emergency Sign Recognition", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
