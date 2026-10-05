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

`GET /api/health` reports each model independently. The alphabet and fingerspelling models depend only on their bundled `.keras` files and labels. The word model additionally checks that its classifier, labels, and normalization stats are deployed, that the I3D checkpoint exists and is not a Git LFS pointer (pointer files are detected by their `version https://git-lfs` header), and that PyTorch and OpenCV are importable. Failures surface as actionable messages ("git lfs pull", "pip install -r backend/requirements-word.txt") rather than silent errors.

## Bidirectional interaction

- **Sign to text:** camera → one of the three models → letter or word → editable message → browser speech synthesis.
- **Text to sign guidance:** typed message → ordered alphabet tiles. This is a fingerspelling sequence, not generated sign-language video.

## Deployment boundary

The default Docker image bundles the alphabet and fingerspelling models — a few MB with no extra system dependencies. Building with `WITH_WORD_MODEL=true` adds the CISLR classifier, PyTorch (CPU wheels), OpenCV, and downloads the 57 MB I3D checkpoint at build time. Continuous word recognition is considerably heavier to run (a full I3D forward pass per clip) and should eventually be split into a separate inference service with its own scaling rules.
