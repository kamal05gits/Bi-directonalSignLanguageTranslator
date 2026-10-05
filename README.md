# SignBridge — Bidirectional ISL Translator

SignBridge is a camera-based Indian Sign Language (ISL) translation prototype with **three bundled recognition models**. It recognizes alphabet signs from webcam photos, reads finger positions from your hand skeleton, classifies short signed-word video clips, lets you assemble a message, reads it aloud, and turns typed text into an easy-to-follow fingerspelling sequence.

## What works

- **Three recognition models (in `backend/models/`)**
  - **Alphabet photo model** — 26-class CNN on a single 64×64 camera frame
  - **Fingerspelling landmark model** — 26-class MLP on a 126-value hand-skeleton vector extracted in the browser with MediaPipe (background independent)
  - **CISLR word model** — 82-class word classifier over I3D video features from a ~3-second clip (requires the Git LFS checkpoint and optional PyTorch/OpenCV dependencies)
- Browser webcam capture with front/rear camera switching and 3-second clip recording
- Confidence thresholds and ranked alternative predictions for every model
- Model picker UI with per-model availability indicators fed by `/api/health`
- Message editing, browser text-to-speech, and text-to-letter sequencing
- Responsive, accessible frontend served by the same FastAPI service
- Docker and Render Blueprint deployment
- Health endpoint and lazy model loading for reliable deploys

> **Scope:** The alphabet and fingerspelling models deploy everywhere with the base image. The word model needs the 57 MB I3D checkpoint (stored with Git LFS) plus PyTorch and OpenCV, so it ships behind a Docker build flag and degrades gracefully: when its pieces are missing, `/api/health` and the UI explain exactly what to install instead of returning simulated predictions.

## Architecture

```text
Browser webcam
   │ JPEG frame ──────────────► POST /api/predict/alphabet ─► Keras CNN (26 letters)
   │ MediaPipe landmarks ─────► POST /api/predict/fingerspelling ─► Keras MLP (26 letters)
   │ 3 s WebM/MP4 clip ───────► POST /api/predict/word ─► I3D features ─► CISLR classifier (82 words)
   ▼
Confidence + alternatives
   ▼
Message builder / speech / fingerspelling sequence
```

The frontend uses relative API URLs, so it works locally and on Render without CORS configuration. Images, landmark vectors, and clips are processed in memory and never written to disk (the word predictor uses a self-deleting temp file purely so OpenCV can decode the clip).

## Run locally with Docker

```bash
# Lean image: alphabet + fingerspelling models
docker build -t signbridge .
docker run --rm -p 10000:10000 signbridge

# Full image: all three models, including the CISLR word model
docker build --build-arg WITH_WORD_MODEL=true -t signbridge:full .
docker run --rm -p 10000:10000 signbridge:full
```

Open <http://localhost:10000>. Browsers permit camera access on localhost. On a remote host, HTTPS is required; Render supplies HTTPS automatically.

## Run locally with Python

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 10000
```

This serves the alphabet and fingerspelling models. To also enable the word model:

```bash
# 1. Restore the real 57 MB I3D checkpoint (the repository holds an LFS pointer)
git lfs pull

# 2. Install PyTorch and OpenCV (CPU wheels keep the download small)
pip install -r backend/requirements-word.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

`GET /api/health` then reports `"word": {"available": true, ...}` and the Word tab in the UI lights up.

API documentation is available at <http://localhost:10000/docs>.

## Deploy to Render

This repository includes `render.yaml` and a production `Dockerfile`.

1. Push the repository to GitHub.
2. In the [Render dashboard](https://dashboard.render.com/), choose **New → Blueprint**.
3. Connect this repository and approve the `signbridge-isl-translator` service.
4. Use at least the **Starter** instance. TensorFlow commonly exceeds the memory available on free instances.
5. Wait for the Docker build and open the generated `onrender.com` URL.
6. Visit `/api/health`; it should return `status: "ok"` with the alphabet and fingerspelling models `available: true`.
7. Make the first prediction. The first request can take longer because the model loads lazily.

The default Render deployment is the lean build. To deploy the word model as well, change the Dockerfile default to `ARG WITH_WORD_MODEL=true` before deploying and choose at least the **Standard** plan — TensorFlow plus PyTorch needs more memory than Starter provides. The Dockerfile downloads the I3D checkpoint during the build, so no LFS setup is needed on Render.

Render reads the `PORT` environment variable automatically. The container runs one worker to avoid loading multiple copies of TensorFlow into memory.

## API

### `GET /api/health`

Reports service and per-model availability without loading TensorFlow. The top-level `model_available`/`model_loaded` fields describe the alphabet model for backward compatibility; the `models` map covers all three.

### `GET /api/info`

Returns supported labels and capture guidance for every model.

### `POST /api/predict` (alias) and `POST /api/predict/alphabet`

Send multipart form data with an image field named `file`. JPEG, PNG, and WebP up to 5 MB are accepted.

```bash
curl -F "file=@hand-sign.jpg" http://localhost:10000/api/predict/alphabet
```

Example response:

```json
{
  "label": "a",
  "confidence": 0.94,
  "accepted": true,
  "top_predictions": [
    {"label": "a", "confidence": 0.94},
    {"label": "s", "confidence": 0.03},
    {"label": "e", "confidence": 0.01}
  ]
}
```

### `POST /api/predict/fingerspelling`

Send JSON with a 126-value `landmarks` array: 2 hands × 21 points × xyz, wrist-relative, left hand in slots 0–62 and right hand in slots 63–125 (the same layout as `backend/app/vision/landmarks.py`). The web UI extracts this vector in the browser with MediaPipe.

```bash
curl -H "Content-Type: application/json" \
  -d '{"landmarks": [0.0, 0.1, 0.0, "... 126 values ..."]}' \
  http://localhost:10000/api/predict/fingerspelling
```

### `POST /api/predict/word`

Send multipart form data with a video field named `file`: a ~3-second WebM, MP4, MOV, or AVI clip of one sign, up to 32 MB. Requires the I3D checkpoint and the optional word dependencies; returns HTTP 503 with an actionable message when they are absent.

```bash
curl -F "file=@sign.webm" http://localhost:10000/api/predict/word
```

## Configuration

| Environment variable | Default | Purpose |
|---|---:|---|
| `PORT` | `10000` | HTTP listen port |
| `CONFIDENCE_THRESHOLD` | `0.70` | Minimum confidence for the alphabet and fingerspelling models |
| `WORD_CONFIDENCE_THRESHOLD` | `0.30` | Minimum confidence for the word model |
| `MAX_UPLOAD_BYTES` | `5242880` | Maximum input image size |
| `MAX_VIDEO_BYTES` | `33554432` | Maximum input video size |
| `ALPHABET_MODEL_PATH` / `ALPHABET_LABELS_PATH` | bundled model | Override alphabet artifacts |
| `FINGERSPELLING_MODEL_PATH` / `FINGERSPELLING_LABELS_PATH` | bundled model | Override fingerspelling artifacts |
| `WORD_MODEL_PATH` / `WORD_LABELS_PATH` / `WORD_NORMALIZATION_PATH` | bundled model | Override word classifier artifacts |
| `I3D_WEIGHTS_PATH` / `I3D_CODE_DIR` | bundled paths | Override I3D checkpoint and module location |

## Privacy and limitations

- Captured images, landmark vectors, and clips are processed in memory and are not written to disk.
- Recognition quality depends on lighting, background, camera framing, and how closely signs match the training data.
- Some letters require motion in certain sign-language conventions; single-frame models cannot represent that movement.
- The word model recognizes the 82 CISLR classes it was trained on, with modest top-1 accuracy — treat its output as a suggestion, not a transcription.
- This is an assistive prototype, not a substitute for a qualified interpreter in medical, legal, or emergency contexts. See [docs/limitations.md](docs/limitations.md).

## Tests

```bash
python -m pytest -q
```

The suite covers dataset importers, feature extraction, prediction engines, the three model services, and the HTTP API. Tests that need FastAPI, TensorFlow, PyTorch, or OpenCV skip automatically when those packages are not installed.
