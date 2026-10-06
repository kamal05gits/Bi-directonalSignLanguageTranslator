FROM python:3.11-slim

# Build with --build-arg WITH_WORD_MODEL=true to include the CISLR word model
# (PyTorch + OpenCV + the 57 MB I3D checkpoint). The default lean image serves
# the alphabet photo model and the hand-landmark fingerspelling model.
ARG WITH_WORD_MODEL=true

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_NUM_INTRAOP_THREADS=2 \
    TF_NUM_INTEROP_THREADS=2 \
    PORT=10000

WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

RUN if [ "$WITH_WORD_MODEL" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends curl && \
      rm -rf /var/lib/apt/lists/* && \
      pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir opencv-python-headless==4.10.0.84; \
    fi

COPY backend /app/backend
COPY frontend /app/frontend

ARG I3D_WEIGHTS_URL=https://media.githubusercontent.com/media/kamal05gits/Bi-directonalSignLanguageTranslator/main/backend/i3d/weights/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt
RUN if [ "$WITH_WORD_MODEL" = "true" ]; then \
      weights="/app/backend/i3d/weights/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt" && \
      mkdir -p "$(dirname "$weights")" && \
      { [ -s "$weights" ] && ! head -c 8 "$weights" | grep -q "^version "; } || \
        curl -fSL "$I3D_WEIGHTS_URL" -o "$weights"; \
    else \
      rm -rf /app/backend/i3d /app/backend/models/cislr; \
    fi

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 10000
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
