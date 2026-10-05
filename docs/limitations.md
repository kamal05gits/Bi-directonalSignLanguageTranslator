# Limitations

SignBridge is an assistive prototype. These constraints are displayed honestly in the product rather than hidden.

## Alphabet photo model

- Recognizes static, single-frame alphabet signs only. Letters that involve motion in ISL (such as J and Z conventions that draw shapes) cannot be represented by a 64×64 photo classifier.
- Sensitive to lighting, background clutter, skin-tone differences in the training data, and camera framing.

## Fingerspelling landmark model

- Depends on the MediaPipe hand tracker running in the browser; when the tracker fails (occlusion, fast motion, low light), no landmarks are sent and the UI asks the user to retry instead of guessing.
- Assumes the same wrist-relative coordinate conventions as the training vectors (`backend/app/vision/landmarks.py`).
- The MediaPipe library loads from a CDN the first time this mode is used; the mode is unavailable fully offline.

## CISLR word model

- Vocabulary is limited to the 82 CISLR classes bundled in `CISLR_LABELS.json`.
- Known modest top-1 accuracy from the underlying I3D checkpoint; predictions are suggestions, not transcriptions.
- Each request runs a full I3D forward pass over 90 frames — expect several seconds of latency on a small CPU instance.
- Requires the 57 MB Git LFS checkpoint plus PyTorch and OpenCV. When any piece is missing, `/api/health` marks the model unavailable and `/api/predict/word` returns HTTP 503 with the remediation step. It never fabricates a prediction.

## All models

- Recognized text should be verified by the conversation partners, especially for names, numbers, and anything safety-critical.
- Not a substitute for a qualified sign-language interpreter in medical, legal, or emergency contexts.
