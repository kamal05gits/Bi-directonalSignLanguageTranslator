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

## Continuous recognition (`/api/continuous`)

- Builds sentences from isolated per-frame fingerspelling predictions smoothed over a short window; it is sequential isolated-sign recognition, not full continuous sentence-level ISL grammar understanding (no co-articulation or non-manual-marker modeling).
- Inherits the fingerspelling model's accuracy and lighting/background sensitivity, plus added latency from the hold/cooldown smoothing (roughly 1–2 seconds per accepted letter by design, to avoid spamming duplicates).

## Translation (`/api/translate`)

- A bundled, offline dictionary covering English → Tamil and Hindi for a few dozen common words and ten whole phrases - not a general-purpose machine translator. Unresolved words are reported, never guessed or silently dropped.

## Emergency phrases (`/api/emergency`)

- **Prototype only.** It does not call police, an ambulance, or any real dispatch/SMS/telephony service. It only displays and speaks a pre-translated phrase and keeps an in-memory log for demo purposes. Do not rely on it in a genuine emergency.

## All models

- Recognized text should be verified by the conversation partners, especially for names, numbers, and anything safety-critical.
- Not a substitute for a qualified sign-language interpreter in medical, legal, or emergency contexts.
