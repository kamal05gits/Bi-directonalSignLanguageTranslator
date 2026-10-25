/**
 * Signora Camera Module
 * Shared client-side camera with multi-camera support and frame prediction.
 * Usage:
 *   const cam = new SignoraCamera('#video-el', { onPrediction: (data) => {} });
 *   await cam.start();
 */

class SignoraCamera {
    constructor(videoSelector, options = {}) {
        this.video = typeof videoSelector === 'string'
            ? document.querySelector(videoSelector)
            : videoSelector;
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.stream = null;
        this.predicting = false;
        this.predictionInFlight = false;
        this.predictInterval = null;
        this.captureInterval = null;

        // Options
        this.fps = options.fps || 3; // prediction frequency
        this.onPrediction = options.onPrediction || (() => { });
        this.onCameraReady = options.onCameraReady || (() => { });
        this.onError = options.onError || ((e) => console.error('Camera error:', e));
        this.buildSentence = options.buildSentence || false;
        this.quality = options.quality || 0.6;
        this.mirror = options.mirror !== false; // default: mirror
        this.modelId = options.modelId || 'cnn';
    }

    /** Enumerate available cameras and return list */
    async getCameraList() {
        try {
            // Need a temporary stream to get permissions first
            const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
            tempStream.getTracks().forEach(t => t.stop());

            const devices = await navigator.mediaDevices.enumerateDevices();
            return devices
                .filter(d => d.kind === 'videoinput')
                .map((d, i) => ({
                    deviceId: d.deviceId,
                    label: d.label || `Camera ${i + 1}`
                }));
        } catch (e) {
            this.onError(e);
            return [];
        }
    }

    /** Build camera selector dropdown */
    async buildCameraSelector(containerSelector) {
        const container = typeof containerSelector === 'string'
            ? document.querySelector(containerSelector)
            : containerSelector;
        if (!container) return;

        const cameras = await this.getCameraList();
        if (cameras.length <= 1) {
            container.style.display = 'none';
            return;
        }

        const select = document.createElement('select');
        select.className = 'form-select form-select-sm';
        select.id = 'camera-select';
        select.style.cssText = `
            background: var(--bg-elevated, #1a1a1a);
            color: var(--text-primary, #fff);
            border: 1px solid var(--border, #333);
            border-radius: var(--radius-sm, 6px);
            font-size: 0.85em;
            padding: 4px 8px;
        `;

        cameras.forEach(cam => {
            const opt = document.createElement('option');
            opt.value = cam.deviceId;
            opt.textContent = cam.label;
            select.appendChild(opt);
        });

        select.addEventListener('change', () => this.switchCamera(select.value));

        const label = document.createElement('span');
        label.textContent = '📷 ';
        label.style.cssText = 'font-size:0.85em; color: var(--text-muted, #888);';

        container.innerHTML = '';
        container.appendChild(label);
        container.appendChild(select);
    }

    /** Start camera with optional deviceId */
    async start(deviceId = null) {
        try {
            if (this.stream) this.stop();

            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            };
            if (deviceId) {
                constraints.video = { deviceId: { exact: deviceId }, width: { ideal: 640 }, height: { ideal: 480 } };
            }

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;
            await this.video.play();

            if (this.mirror) {
                this.video.style.transform = 'scaleX(-1)';
            }

            this.onCameraReady();
            return true;
        } catch (e) {
            this.onError(e);
            return false;
        }
    }

    /** Switch to a different camera */
    async switchCamera(deviceId) {
        const wasPredicting = this.predicting;
        if (wasPredicting) this.stopPredicting();
        await this.start(deviceId);
        if (wasPredicting) this.startPredicting();
    }

    /** Stop camera */
    stop() {
        this.stopPredicting();
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
            this.stream = null;
        }
    }

    /** Capture current frame as base64 JPEG */
    captureFrame() {
        if (!this.video || !this.video.videoWidth) return null;
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        this.ctx.drawImage(this.video, 0, 0);
        return this.canvas.toDataURL('image/jpeg', this.quality);
    }

    /** Send frame to server for prediction */
    async predict() {
        if (this.predictionInFlight) return null;
        const frame = this.captureFrame();
        if (!frame) return null;

        this.predictionInFlight = true;
        try {
            const t0 = performance.now();
            const r = await fetch('/api/predict_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: frame,
                    build_sentence: this.buildSentence,
                    model_id: this.modelId
                })
            });
            let data;
            try {
                data = await r.json();
            } catch (error) {
                throw new Error(`Prediction service returned an invalid response (HTTP ${r.status}).`);
            }
            if (!data || typeof data !== 'object') {
                throw new Error(`Prediction service returned an invalid response (HTTP ${r.status}).`);
            }
            if (!r.ok && !data.error) data.error = `Prediction failed (${r.status})`;
            data._latency = Math.round(performance.now() - t0);
            this.onPrediction(data);
            return data;
        } catch (e) {
            this.onPrediction({
                error: e.message || 'Prediction request failed.',
                model_id: this.modelId
            });
            console.error('Prediction request failed:', e);
            return null;
        } finally {
            this.predictionInFlight = false;
        }
    }

    /** Start continuous prediction at configured FPS */
    startPredicting() {
        if (this.predicting) return;
        this.predicting = true;
        this.predictInterval = setInterval(() => this.predict(), 1000 / this.fps);
    }

    /** Adjust prediction cadence without restarting the camera stream */
    setPredictionFps(fps) {
        this.fps = fps;
        if (!this.predicting) return;
        clearInterval(this.predictInterval);
        this.predictInterval = setInterval(() => this.predict(), 1000 / this.fps);
    }

    /** Stop continuous prediction */
    stopPredicting() {
        this.predicting = false;
        if (this.predictInterval) {
            clearInterval(this.predictInterval);
            this.predictInterval = null;
        }
    }

    async waitForPredictionIdle() {
        while (this.predictionInFlight) {
            await new Promise(resolve => setTimeout(resolve, 25));
        }
    }
}

// Export for use
window.SignoraCamera = SignoraCamera;
