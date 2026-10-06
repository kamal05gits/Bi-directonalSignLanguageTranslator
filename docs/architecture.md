# Web architecture

SignBridge is deployed as one Dockerized FastAPI service. FastAPI hosts both the static browser client and the prediction API, keeping camera and API traffic on one HTTPS origin.

## Recognition flows

The service exposes three recognition models, all served from `backend/app/services/` behind a shared lazy-loading base (`keras_classifier.LazyKerasClassifier`):

1. **Alphabet photo model** (`/api/predict/alphabet`, aliased by `/api/predict`)
   - The client crops a square frame, mirrors the selfie camera, and encodes JPEG.
   - Pillow corrects EXIF orientation, converts RGB, and resizes to 64×64.
   - The Keras CNN returns probabilities for 26 alphabet labels.
2. **Fingerspelling landmark model** (`/api/predict/fingerspelling`)
   - The browser runs MediaPipe Hands locally on the raw (unmirrored) frame and derives the 126-value feature vector — 2 hands × 21 landmarks × xyz, wrist-relative, left hand in slots 0–62 — identical to `app.vision.landmarks.extract_landmark_features`.
   - Only the coordinates are uploaded; the Keras MLP (with its own adapted Normalization layer) returns 26 letter probabilities.
3. **CISLR word model** (`/api/predict/word`)
   - The browser records a ~3-second clip (MediaRecorder, WebM/MP4) and uploads it.
   - The server decodes frames with OpenCV, resamples to 90 frames at 224×224, extracts 1024-d features with the I3D network (`backend/i3d/pytorch_i3d.py`, asl2000 checkpoint), resamples to (32, 1024), applies the training normalization, and classifies into 82 word labels.

For every model, the API returns the top label, its confidence, acceptance against a threshold, and ranked alternatives. The browser only enables **Add letter**/**Add word** above the confidence threshold.

All models are loaded lazily and protected by locks. Render health checks therefore do not wait for TensorFlow startup, and concurrent requests do not invoke the models unsafely. One Uvicorn worker prevents duplicate in-memory model copies.

## Model availability reporting

`GET /api/health` reports each model independently. The alphabet and fingerspelling models depend only on their bundled `.keras` files and labels. The word model additionally checks that its classifier, labels, and normalization stats are deployed and that PyTorch and OpenCV are importable. If its I3D checkpoint is a Git LFS pointer, it is downloaded on the first prediction, checked against the pinned SHA-256, and atomically installed; malformed, oversized, partial, or checksum-mismatched downloads are discarded. Automatic download can be disabled with `WORD_AUTO_DOWNLOAD=0`. Failures surface as actionable messages rather than simulated predictions.

At model load and prediction time, every classifier also validates its output width against the ordered label file and rejects duplicate/empty labels, non-finite values, logits, and malformed probability sums. This prevents a model artifact mismatch from quietly attaching the wrong label or confidence to an output index.

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
  (`continuous.get_predictor`), so tests override it with a lightweight stub instead of needing TensorFlow.
- **`app.routes.language`** (`/api/translate`, `/api/languages`) exposes `DictionaryTranslator` — a bundled,
  fully offline phrase/word dictionary (English → Tamil/Hindi). It never calls a third-party translation
  service (there is no API key in this deployment), and it reports unresolved words explicitly instead of
  guessing, matching the "never fabricate a result" policy used by the recognition models.
- **`app.routes.emergency`** (`/api/emergency/*`) exposes `app.services.emergency` — a small set of
  pre-translated, high-value phrases plus an in-memory alert log. This is explicitly a **prototype**: it is
  not connected to any telephony, SMS, or dispatch provider, and every response says so.

## Bidirectional interaction

- **Sign to text (one shot):** camera → one of the three models → letter or word → editable message → browser speech synthesis.
- **Sign to text (continuous):** camera → streamed landmarks → `/api/continuous` session (buffer + stabilizer + sentence builder) → live-updating editable message.
- **Text to sign guidance:** typed message → ordered alphabet tiles. This is a fingerspelling sequence, not generated sign-language video. Both letter models cover a–z only, so digits, punctuation, and accented characters are rendered as explicit "no letter sign" tiles and listed under the sequence rather than silently skipped.
- **Text to another language:** typed/recognized message → `/api/translate` → original + translated text → speech synthesis in the matching language/voice when available.

## Deployment boundary

The default Docker image bundles the alphabet and fingerspelling models — a few MB with no extra system dependencies. Building with `WITH_WORD_MODEL=true` adds the CISLR classifier, PyTorch (CPU wheels), OpenCV, and downloads the 57 MB I3D checkpoint at build time. Continuous word recognition is considerably heavier to run (a full I3D forward pass per clip) and should eventually be split into a separate inference service with its own scaling rules.
