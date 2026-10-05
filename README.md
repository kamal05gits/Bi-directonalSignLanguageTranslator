# SignBridge — Bidirectional ISL Fingerspelling Translator

SignBridge is a camera-based Indian Sign Language (ISL) fingerspelling prototype. It recognizes a single alphabet hand sign from a webcam image, lets the user assemble letters into a message, reads the message aloud, and turns typed text into an easy-to-follow fingerspelling sequence.

## What works

- Browser webcam capture with front/rear camera switching
- Server-side inference using the included 26-class Keras alphabet model
- Confidence threshold and top-three alternatives
- Message editing, browser text-to-speech, and text-to-letter sequencing
- Responsive, accessible frontend served by the same FastAPI service
- Docker and Render Blueprint deployment
- Health endpoint and lazy model loading for reliable deploys

> **Scope:** The deployed MVP recognizes static, single-frame alphabet signs. The CISLR word model needs a large I3D video feature extractor and checkpoint and is not enabled in the web deployment. This limitation is shown honestly rather than returning simulated predictions.

## Architecture

```text
Browser webcam
   │ JPEG capture (not stored)
   ▼
POST /api/predict
   │ Pillow: RGB + 64×64 resize
   ▼
Keras alphabet CNN (26 labels)
   │ confidence + alternatives
   ▼
Message builder / speech / fingerspelling sequence
```

The frontend uses relative API URLs, so it works locally and on Render without CORS configuration.

## Run locally with Docker

```bash
docker build -t signbridge .
docker run --rm -p 10000:10000 signbridge
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

API documentation is available at <http://localhost:10000/docs>.

## Deploy to Render

This repository includes `render.yaml` and a production `Dockerfile`.

1. Push the repository to GitHub.
2. In the [Render dashboard](https://dashboard.render.com/), choose **New → Blueprint**.
3. Connect this repository and approve the `signbridge-isl-translator` service.
4. Use at least the **Starter** instance. TensorFlow commonly exceeds the memory available on free instances.
5. Wait for the Docker build and open the generated `onrender.com` URL.
6. Visit `/api/health`; it should return `status: "ok"` and `model_available: true`.
7. Make the first prediction. The first request can take longer because the model loads lazily.

Render reads the `PORT` environment variable automatically. The container runs one worker to avoid loading multiple copies of TensorFlow into memory.

## API

### `GET /api/health`

Reports service and model availability without loading TensorFlow.

### `GET /api/info`

Returns supported labels and capture guidance.

### `POST /api/predict`

Send multipart form data with an image field named `file`. JPEG, PNG, and WebP up to 5 MB are accepted.

```bash
curl -F "file=@hand-sign.jpg" http://localhost:10000/api/predict
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

## Configuration

| Environment variable | Default | Purpose |
|---|---:|---|
| `PORT` | `10000` | HTTP listen port |
| `CONFIDENCE_THRESHOLD` | `0.70` | Minimum confidence for adding a result |
| `MAX_UPLOAD_BYTES` | `5242880` | Maximum input image size |
| `ALPHABET_MODEL_PATH` | bundled model | Override model path |
| `ALPHABET_LABELS_PATH` | bundled labels | Override labels path |

## Privacy and limitations

- Captured images are processed in memory and are not written to disk.
- Recognition quality depends on lighting, background, camera framing, and how closely signs match the training data.
- Some letters require motion in certain sign-language conventions; a static-image model cannot represent that movement.
- This is an assistive prototype, not a substitute for a qualified interpreter in medical, legal, or emergency contexts.
