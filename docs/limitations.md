# Limitations

SignBridge is an assistive prototype. These constraints are displayed honestly in the product rather than hidden.

## Alphabet photo model

- Recognizes static, single-frame alphabet signs only. Letters that involve motion in ISL (such as J and Z conventions that draw shapes) cannot be represented by a photo classifier.
- Sensitive to lighting, background clutter, skin-tone differences in the training data, and camera framing.
- Classifier is a `RandomForestClassifier` over the flattened 32×32×3 pixel vector (`backend/app/services/alphabet_predictor.py`).

## Fingerspelling landmark model

- Depends on the MediaPipe hand tracker running in the browser; when the tracker fails (occlusion, fast motion, low light), no landmarks are sent and the UI asks the user to retry instead of guessing.
- Assumes the same wrist-relative coordinate conventions as the training vectors (`backend/app/vision/landmarks.py`).
- The MediaPipe library loads from a CDN the first time this mode is used; the mode is unavailable fully offline.
- Classifier is a `RandomForestClassifier` over the 126-value landmark vector directly (no feature extraction network).

## CISLR word model

- Vocabulary is limited to the 82 CISLR classes bundled in `CISLR_LABELS.json`.
- Known modest top-1 accuracy from the underlying I3D checkpoint; predictions are suggestions, not transcriptions.
- Each request runs a full I3D forward pass over 90 frames — expect several seconds of latency on a small CPU instance.
- Requires the 57 MB Git LFS checkpoint plus PyTorch and OpenCV for I3D feature extraction. When any piece is missing, `/api/health` marks the model unavailable and `/api/predict/word` returns HTTP 503 with the remediation step. It never fabricates a prediction.
- The final classifier is a `RandomForestClassifier` over the pooled (mean + std) I3D feature vector, not a neural network.

## All three Random Forest models

- **None ships with a real training dataset.** `backend/app/ml/train_alphabet_rf.py`, `train_fingerspelling_rf.py`, and `train_word_rf.py` train on a small synthetic placeholder dataset (`app.ml.synthetic_data`) by default so the API has a working model out of the box - it does **not** reflect real hands, photos, or signed words. Pass `--images-dir`, `--landmarks-csv`, or `--features-csv` with real data and retrain before relying on these models for real recognition.

## Continuous recognition (`/api/continuous`)

- Builds sentences from isolated per-frame fingerspelling predictions smoothed over a short window; it is sequential isolated-sign recognition, not full continuous sentence-level ISL grammar understanding (no co-articulation or non-manual-marker modeling).
- Inherits the fingerspelling model's accuracy and lighting/background sensitivity, plus added latency from the hold/cooldown smoothing (roughly 1–2 seconds per accepted letter by design, to avoid spamming duplicates).

## Translation (`/api/translate`)

- A bundled, offline dictionary covering English → Tamil and Hindi for a few dozen common words and ten whole phrases - not a general-purpose machine translator. Unresolved words are reported, never guessed or silently dropped.

## Emergency phrases (`/api/emergency`)

- Displays and speaks a pre-translated phrase and keeps an in-memory log of every alert.
- **Real dispatch is optional and limited.** When `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, and `TWILIO_ALERT_TO_NUMBER` are configured, raising an alert sends a real SMS and places a real voice call via Twilio - but only to one fixed, operator-configured contact number, not to police or an ambulance service directly, and with no location data. Without those variables, it is a **prototype only**: no telephony, SMS, or dispatch provider is contacted. Either way, do not rely on this as a substitute for dialing your local emergency number.

## All models

- Recognized text should be verified by the conversation partners, especially for names, numbers, and anything safety-critical.
- Not a substitute for a qualified sign-language interpreter in medical, legal, or emergency contexts.
