# Web architecture

SignBridge is deployed as one Dockerized FastAPI service. FastAPI hosts both the static browser client and the prediction API, keeping camera and API traffic on one HTTPS origin.

## Recognition flows

The service exposes three recognition models, all served from `backend/app/services/` behind a shared lazy-loading base (`random_forest_classifier.LazyRandomForestClassifier`). Every model's final classifier is a scikit-learn `RandomForestClassifier` serialized with `joblib` - there is no TensorFlow/Keras dependency left in this project:

1. **Alphabet photo model** (`/api/predict/alphabet`, aliased by `/api/predict`)
   - The client crops a square frame, mirrors the selfie camera, and encodes JPEG.
   - Pillow corrects EXIF orientation, converts RGB, and resizes to 64×64.
   - A Random Forest trained on the flattened 32×32×3 pixel vector returns probabilities for 26 alphabet labels.
2. **Fingerspelling landmark model** (`/api/predict/fingerspelling`)
   - The browser runs MediaPipe Hands locally on the raw (unmirrored) frame and derives the 126-value feature vector — 2 hands × 21 landmarks × xyz, wrist-relative, left hand in slots 0–62 — identical to `app.vision.landmarks.extract_landmark_features`.
   - Only the coordinates are uploaded; a Random Forest trained directly on the 126-value vector returns 26 letter probabilities.
3. **CISLR word model** (`/api/predict/word`)
   - The browser records a ~3-second clip (MediaRecorder, WebM/MP4) and uploads it.
   - The server decodes frames with OpenCV, resamples to 90 frames at 224×224, extracts 1024-d features with the I3D network (`backend/i3d/pytorch_i3d.py`, asl2000 checkpoint), resamples to (32, 1024), applies the training normalization, pools the sequence over time into one 2048-value vector (mean + standard deviation per channel, `word_predictor.pool_features`), and classifies that vector with a Random Forest into 82 word labels. PyTorch/OpenCV are still required for the I3D feature-extraction step; only the final classifier changed from a neural network to a Random Forest.

For every model, the API returns the top label, its confidence, acceptance against a threshold, and ranked alternatives. The browser only enables **Add letter**/**Add word** above the confidence threshold.

All models are loaded lazily and protected by locks. Render health checks therefore do not wait for model/I3D startup, and concurrent requests do not invoke the models unsafely. One Uvicorn worker prevents duplicate in-memory model copies.

## Model availability reporting

`GET /api/health` reports each model independently. The alphabet and fingerspelling models depend only on their bundled `.joblib` Random Forest files and labels. The word model additionally checks that its classifier, labels, and normalization stats are deployed, that the I3D checkpoint exists and is not a Git LFS pointer (pointer files are detected by their `version https://git-lfs` header), and that PyTorch and OpenCV are importable. Failures surface as actionable messages ("git lfs pull", "pip install -r backend/requirements-word.txt") rather than silent errors.

### Training the Random Forest models

`backend/app/ml/train_alphabet_rf.py`, `train_fingerspelling_rf.py`, and `train_word_rf.py` (re)train each `.joblib` model. None of the three ships a raw dataset in this repository, so by default each script generates a small synthetic placeholder dataset (`app.ml.synthetic_data`) just so the pipeline and the bundled API work out of the box - it is **not** real sign-language data. Pass `--images-dir` (alphabet, one label subfolder per class), `--landmarks-csv` (fingerspelling, `label` + 126 numeric columns), or `--features-csv` (word, `label` + 2048 pooled-I3D-feature columns) to train on real data instead; see each script's module docstring for the exact format.

## Continuous recognition, NLP, translation, and emergency phrases

Four previously standalone, unit-tested-only components (`app.features.sequence_buffer.SequenceBuffer`,
`app.ml.predict.LabelStabilizer`, `app.language.sentence_processor.SentenceProcessor`, and
`app.language.translator.DictionaryTranslator`) are wired into real, callable API routers under
`backend/app/routes/`:

- **`app.routes.continuous`** (`/api/continuous/*`) holds one in-memory session per active live-recognition
  user: a `SequenceBuffer` averages the last few raw landmark frames (temporal smoothing of the signal
  itself), a `LabelStabilizer` then requires a label to win several consecutive frames above the confidence
  threshold before accepting it and suppresses the same label again for a cooldown window (temporal
  smoothing + duplicate prevention at the label level), and accepted letters are appended to a
  `SentenceProcessor` that supports manual backspace, punctuation, and space insertion. The browser decides
  when to call `/space` (after a few consecutive "no hand" frames), keeping the server stateless beyond the
  session dict. The fingerspelling predictor is supplied through a FastAPI dependency
  (`continuous.get_predictor`), so tests override it with a lightweight stub instead of needing scikit-learn.
- **`app.routes.language`** (`/api/translate`, `/api/languages`) exposes `DictionaryTranslator` — a bundled,
  fully offline phrase/word dictionary (English → Tamil/Hindi). It never calls a third-party translation
  service (there is no API key in this deployment), and it reports unresolved words explicitly instead of
  guessing, matching the "never fabricate a result" policy used by the recognition models.
- **`app.routes.emergency`** (`/api/emergency/*`) exposes `app.services.emergency` — a small set of
  pre-translated, high-value phrases plus an in-memory alert log — and, when configured,
  `app.services.notifications.TwilioNotifier`. Setting `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`,
  `TWILIO_FROM_NUMBER`, and `TWILIO_ALERT_TO_NUMBER` makes `POST /api/emergency/alert` send a real SMS and
  place a real voice call (Twilio `<Say>` text-to-speech) to one fixed emergency contact number. Without
  those variables set, it stays the original **prototype**: no telephony, SMS, or dispatch provider is
  contacted, and every response says so plainly. `GET /api/emergency/status` reports whether Twilio is
  configured without leaking the credentials.

## Bidirectional interaction

- **Sign to text (one shot):** camera → one of the three models → letter or word → editable message → browser speech synthesis.
- **Sign to text (continuous):** camera → streamed landmarks → `/api/continuous` session (buffer + stabilizer + sentence builder) → live-updating editable message.
- **Text to sign guidance:** typed message → ordered alphabet tiles. This is a fingerspelling sequence, not generated sign-language video.
- **Text to another language:** typed/recognized message → `/api/translate` → original + translated text → speech synthesis in the matching language/voice when available.

## Deployment boundary

The default Docker image bundles the alphabet and fingerspelling models — a few MB with no extra system dependencies. Building with `WITH_WORD_MODEL=true` adds the CISLR classifier, PyTorch (CPU wheels), OpenCV, and downloads the 57 MB I3D checkpoint at build time. Continuous word recognition is considerably heavier to run (a full I3D forward pass per clip) and should eventually be split into a separate inference service with its own scaling rules.
