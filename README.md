# SignBridge — Bidirectional ISL Translator

SignBridge is a camera-based Indian Sign Language (ISL) translation prototype with **three bundled Random Forest recognition models and a continuous live-recognition pipeline**. It recognizes alphabet signs from webcam photos, reads finger positions from your hand skeleton (one shot or streamed continuously into a sentence), classifies short signed-word video clips, lets you assemble and edit a message, translates it into Tamil/Hindi, reads it aloud (including in the translated language), surfaces one-tap emergency phrases that can send a real Twilio SMS/call, and turns typed text into an easy-to-follow fingerspelling sequence.

## What works

- **Three recognition models (in `backend/models/`), all scikit-learn Random Forests**
  - **Alphabet photo model** — 26-class Random Forest on a flattened 32×32 camera frame
  - **Fingerspelling landmark model** — 26-class Random Forest on a 126-value hand-skeleton vector extracted in the browser with MediaPipe (background independent)
  - **CISLR word model** — 82-class Random Forest over pooled I3D video features (mean + std of 1024-d features across the clip, requires the Git LFS checkpoint and optional PyTorch/OpenCV dependencies for the I3D feature extractor)
  - None of the three ships a raw training dataset, so each is trained on a small synthetic placeholder dataset by default (`backend/app/ml/train_*_rf.py` + `app.ml.synthetic_data`) — retrain with `--images-dir`/`--landmarks-csv`/`--features-csv` on real data for real recognition accuracy.
- **Continuous recognition mode** (`/api/continuous/*`) — streams landmark frames from the browser into a server-side session that smooths noisy per-frame predictions (`app.ml.predict.LabelStabilizer`), averages a rolling landmark buffer (`app.features.sequence_buffer.SequenceBuffer`), and assembles accepted letters into editable text with the sentence/NLP layer (`app.language.sentence_processor.SentenceProcessor`) — including automatic space insertion on a hand-away pause and manual backspace/punctuation.
- **Multilingual translation** (`/api/translate`, `/api/languages`) — a bundled, fully-offline dictionary/phrase translator (English → Tamil/Hindi today) that reports unresolved words explicitly instead of guessing.
- **Emergency phrases** (`/api/emergency/*`) — one-tap, pre-translated high-value phrases (help, ambulance, police, doctor, deaf, fire, lost, hospital) shown large on screen and spoken aloud immediately. When `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, and `TWILIO_ALERT_TO_NUMBER` are configured, raising an alert also sends a **real SMS and places a real voice call** via Twilio to one fixed emergency contact number. Without those variables set, it stays a **prototype**: no telephony, SMS, or dispatch provider is contacted, and every response says so.
- Browser webcam capture with front/rear camera switching, pause/resume, and 3-second clip recording
- Live hand-landmark overlay drawn over the video feed (MediaPipe drawing utils)
- Confidence thresholds and ranked alternative predictions for every model
- Model picker UI with per-model availability indicators fed by `/api/health`
- Message editing, browser text-to-speech with replay/stop/mute and per-language voice selection, and text-to-letter sequencing
- Responsive, accessible frontend served by the same FastAPI service
- Docker and Render Blueprint deployment
- Health endpoint and lazy model loading for reliable deploys

> **Scope:** The alphabet and fingerspelling models deploy everywhere with the base image. The word model needs the 57 MB I3D checkpoint (stored with Git LFS) plus PyTorch and OpenCV, so it ships behind a Docker build flag and degrades gracefully: when its pieces are missing, `/api/health` and the UI explain exactly what to install instead of returning simulated predictions.

## Architecture

```text
Browser webcam
   │ JPEG frame ──────────────► POST /api/predict/alphabet ─► Random Forest (26 letters)
   │ MediaPipe landmarks ─────► POST /api/predict/fingerspelling ─► Random Forest (26 letters)
   │ 3 s WebM/MP4 clip ───────► POST /api/predict/word ─► I3D features ─► pooled ─► Random Forest (82 words)
   ▼
Confidence + alternatives
   ▼
Message builder / speech / fingerspelling sequence

One-tap emergency phrase ──► POST /api/emergency/alert ─► Twilio SMS + voice call (if configured)
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
4. Use at least the **Starter** instance; the Random Forest models are lightweight compared to the previous neural networks.
5. Wait for the Docker build and open the generated `onrender.com` URL.
6. Visit `/api/health`; it should return `status: "ok"` with the alphabet and fingerspelling models `available: true`.
7. Make the first prediction. The first request can take longer because the model loads lazily.

The default Render deployment is the lean build. To deploy the word model as well, change the Dockerfile default to `ARG WITH_WORD_MODEL=true` before deploying and choose at least the **Standard** plan — PyTorch (for I3D feature extraction) needs more memory than Starter provides. The Dockerfile downloads the I3D checkpoint during the build, so no LFS setup is needed on Render.

To enable real emergency SMS/voice-call dispatch, set `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, and `TWILIO_ALERT_TO_NUMBER` in the Render dashboard's Environment tab (see `.env.example`); `render.yaml` declares them as `sync: false` placeholders so no secret is ever committed.

Render reads the `PORT` environment variable automatically. The container runs one worker to avoid loading multiple copies of each model into memory.

## API

### `GET /api/health`

Reports service and per-model availability without loading scikit-learn or PyTorch. The top-level `model_available`/`model_loaded` fields describe the alphabet model for backward compatibility; the `models` map covers all three.

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

### Continuous recognition: `POST /api/continuous/session` + `.../frame`

Create a session, then stream one 126-value landmark vector per frame (an all-zero vector means "no hand visible"). The server averages a short rolling buffer, requires a label to hold for several consecutive frames before accepting it, and suppresses immediate repeats — only then does it append the letter to the session's sentence.

```bash
SID=$(curl -s -X POST http://localhost:10000/api/continuous/session | python3 -c "import json,sys;print(json.load(sys.stdin)['session_id'])")
curl -H "Content-Type: application/json" -d '{"landmarks": [ ...126 values... ]}' \
  http://localhost:10000/api/continuous/session/$SID/frame
```

Other endpoints on the same session: `GET .../session/{id}` (current text/tokens), `POST .../space`, `POST .../punctuation` (`{"mark": "."}`), `POST .../backspace`, `POST .../clear`, `DELETE .../session/{id}`.

### `GET /api/languages` and `POST /api/translate`

Lists supported target languages and translates English text/phrases with a bundled offline dictionary (no third-party translation API or network call). Unresolved words are reported explicitly rather than guessed.

```bash
curl -H "Content-Type: application/json" -d '{"text": "thank you", "language": "ta"}' \
  http://localhost:10000/api/translate
```

### `GET /api/emergency/phrases`, `GET /api/emergency/status`, and `POST /api/emergency/alert`

Lists pre-translated high-value phrases and records every raised alert (in memory). When `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, and `TWILIO_ALERT_TO_NUMBER` are set (see [Configuration](#configuration) and `.env.example`), `POST /api/emergency/alert` also sends a **real SMS and places a real voice call** via Twilio to that one fixed emergency contact number, and the response's `dispatched`/`sms_sid`/`call_sid`/`dispatch_errors` fields report what actually happened. `GET /api/emergency/status` reports whether Twilio is configured (without leaking credentials) so the frontend can show the right badge. Without those variables, it stays a **prototype**: no telephony, SMS, or dispatch provider is contacted, and every response states this plainly.

```bash
curl http://localhost:10000/api/emergency/phrases?language=hi
curl http://localhost:10000/api/emergency/status
curl -H "Content-Type: application/json" -d '{"phrase_id": "ambulance", "language": "ta"}' \
  http://localhost:10000/api/emergency/alert
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
| `TWILIO_ACCOUNT_SID` | unset | Twilio Account SID; all four `TWILIO_*` values below must be set to enable real dispatch |
| `TWILIO_AUTH_TOKEN` | unset | Twilio Auth Token |
| `TWILIO_FROM_NUMBER` | unset | Twilio phone number alerts are sent/called from (E.164, e.g. `+15551234567`) |
| `TWILIO_ALERT_TO_NUMBER` | unset | Fixed emergency contact number that receives every alert (E.164) |
| `TWILIO_ENABLE_SMS` / `TWILIO_ENABLE_CALL` | `true` / `true` | Disable one channel while keeping the other |

See `.env.example` for a copy-pasteable template of all of these.

## Privacy and limitations

- Captured images, landmark vectors, and clips are processed in memory and are not written to disk.
- Recognition quality depends on lighting, background, camera framing, and how closely signs match the training data.
- Some letters require motion in certain sign-language conventions; single-frame models cannot represent that movement.
- The word model recognizes the 82 CISLR classes it was trained on, with modest top-1 accuracy — treat its output as a suggestion, not a transcription.
- **None of the three bundled Random Forest models is trained on real sign-language data** — no raw dataset ships in this repository, so `backend/app/ml/train_*_rf.py` fall back to a small synthetic placeholder dataset by default. They prove the training → `.joblib` → API pipeline end-to-end but will not recognize real hands/photos/clips until retrained on real data (`--images-dir` / `--landmarks-csv` / `--features-csv`).
- When Twilio is configured, raising an emergency alert sends the phrase text to Twilio (a third-party service) by SMS/voice call to the configured contact number; no other recognition data (images, landmarks, video) is ever sent anywhere.
- This is an assistive prototype, not a substitute for a qualified interpreter in medical, legal, or emergency contexts. See [docs/limitations.md](docs/limitations.md).

## Tests

```bash
python -m pytest -q
```

The suite covers dataset importers, feature extraction, prediction engines (including the live-stream confidence stabilizer), the three Random Forest model services, the Twilio notification service, the sentence/translation/emergency modules, and the full HTTP API (continuous session, translation, emergency). Tests that need FastAPI, scikit-learn, PyTorch, or OpenCV skip automatically when those packages are not installed.
