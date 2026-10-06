# Setup

See the root [README](../README.md) for Docker, local Python, API, and Render deployment instructions.

The minimum production environment is Python 3.11 with the packages pinned in `backend/requirements.txt`. Camera capture requires either `localhost` or an HTTPS deployment.

## Enabling the CISLR word model

The alphabet and fingerspelling models work with the base requirements. The word model (`POST /api/predict/word`) needs three extra pieces:

1. **The I3D checkpoint.** The repository tracks it with Git LFS, so a plain clone can contain only a small pointer file. SignBridge now downloads the real 57 MB file on the first word prediction, verifies its pinned SHA-256, and installs it atomically. You can still restore it ahead of time:

   ```bash
   git lfs pull
   ```

   Set `WORD_AUTO_DOWNLOAD=0` to require manual provisioning, or use `I3D_WEIGHTS_PATH` for a copy stored elsewhere. Custom downloads require matching `I3D_WEIGHTS_URL` and `I3D_WEIGHTS_SHA256` values.

2. **The optional Python dependencies** (`torch`, `opencv-python-headless`):

   ```bash
   pip install -r backend/requirements-word.txt \
     --extra-index-url https://download.pytorch.org/whl/cpu
   ```

   The service imports both lazily, so the rest of the API runs fine without them.

3. **The CISLR artifacts** (`backend/models/cislr/`): classifier, labels, and normalization stats. They are committed to the repository and included in Docker builds made with `--build-arg WITH_WORD_MODEL=true`.

`GET /api/health` reports exactly which piece is missing. In Docker, `docker build --build-arg WITH_WORD_MODEL=true .` handles all three pieces automatically (the Dockerfile downloads the checkpoint at build time).

## Enabling real emergency SMS + voice calls (Twilio)

Emergency alerts work out of the box as an on-screen/spoken prototype. To deliver
them **for real** to your phone as an SMS and a voice call, set four environment
variables before starting the server:

```bash
export TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # from the Twilio Console
export TWILIO_AUTH_TOKEN=your_auth_token
export TWILIO_FROM_NUMBER=+15550001111                          # your Twilio number (E.164)
export EMERGENCY_TO_NUMBER=+919876543210                        # the number that receives alerts (E.164)
```

Then every `POST /api/emergency/alert` sends an SMS and places a voice call that
speaks the phrase (Hindi voices are used when the phrase was raised in Hindi).
Two optional toggles exist:

- `EMERGENCY_SMS_ENABLED=false` — disable the SMS channel.
- `EMERGENCY_VOICE_CALL_ENABLED=false` — disable the voice-call channel.

If you use a Messaging Service instead of a single sender number, set
`TWILIO_MESSAGING_SERVICE_SID` (SMS only; the voice call still needs
`TWILIO_FROM_NUMBER`).

Verification: `GET /api/emergency/status` reports the provider configuration and
a masked destination number. Alerts notify your configured contact only — this
is not a public emergency-services (112 / 911) integration.

## Combined (all three models) mode

`POST /api/predict/combined` accepts any mix of `image`, `landmarks`, and
`video` in one request and merges the results. The Combined tab in the UI
records one clip, grabs a mid-sign frame plus its landmarks, and sends all
three together; when the word model is unavailable it degrades to the two
letter models automatically, and when no hand is detected it still uses the
photo model.
