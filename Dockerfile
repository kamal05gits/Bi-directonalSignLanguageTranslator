# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p data/raw/gestures logs static/uploads

# Environment variables for cloud mode
ENV USE_SERVER_CAMERA=false
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
