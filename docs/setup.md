# Setup

See the root [README](../README.md) for Docker, local Python, API, and Render deployment instructions.

The minimum production environment is Python 3.11 with the packages pinned in `backend/requirements.txt`. Camera capture requires either `localhost` or an HTTPS deployment.

## Enabling the CISLR word model

The alphabet and fingerspelling models work with the base requirements. The word model (`POST /api/predict/word`) needs three extra pieces:

1. **The I3D checkpoint.** The repository tracks it with Git LFS, so a plain clone contains only a small pointer file. Restore the real 57 MB file:

   ```bash
   git lfs pull
   ```

   The server detects pointer files and reports "I3D checkpoint is a Git LFS pointer" until the real file is present. Alternatively set `I3D_WEIGHTS_PATH` to a copy stored elsewhere.

2. **The optional Python dependencies** (`torch`, `opencv-python-headless`):

   ```bash
   pip install -r backend/requirements-word.txt \
     --extra-index-url https://download.pytorch.org/whl/cpu
   ```

   The service imports both lazily, so the rest of the API runs fine without them.

3. **The CISLR artifacts** (`backend/models/cislr/`): classifier, labels, and normalization stats. They are committed to the repository and included in Docker builds made with `--build-arg WITH_WORD_MODEL=true`.

`GET /api/health` reports exactly which piece is missing. In Docker, `docker build --build-arg WITH_WORD_MODEL=true .` handles all three pieces automatically (the Dockerfile downloads the checkpoint at build time).
