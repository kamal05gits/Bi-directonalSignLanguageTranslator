# Web architecture

SignBridge is deployed as one Dockerized FastAPI service. FastAPI hosts both the static browser client and the prediction API, keeping camera and API traffic on one HTTPS origin.

## Recognition flow

1. The user grants camera permission in the browser.
2. The client crops a square frame and encodes it as JPEG.
3. `POST /api/predict` validates the MIME type and 5 MB size limit.
4. Pillow corrects EXIF orientation, converts RGB, and resizes to 64×64.
5. The Keras CNN returns probabilities for 26 alphabet labels.
6. The API returns the highest-confidence result and two alternatives.
7. The browser only enables **Add letter** above the confidence threshold.

The model is loaded lazily and protected by a lock. Render health checks therefore do not wait for TensorFlow startup, and concurrent requests do not invoke the model unsafely. One Uvicorn worker prevents duplicate in-memory model copies.

## Bidirectional interaction

- **Sign to text:** camera image → classifier → letter → editable message → browser speech synthesis.
- **Text to sign guidance:** typed message → ordered alphabet tiles. This is a fingerspelling sequence, not generated sign-language video.

## Deployment boundary

The production Docker image excludes I3D, CISLR, and auxiliary training artifacts. Continuous word recognition requires the actual Git LFS I3D checkpoint and considerably more CPU/RAM; it should eventually be deployed as a separate inference service.
