/**
 * ISL Gesture Recognition System - Frontend JavaScript
 */

class ISLGestureApp {
    constructor() {
        this.currentSentence = '';
        this.currentGesture = '';
        this.currentConfidence = 0.0;
        this.handCount = 0;
        this.isProcessing = false;
        this.isSpeaking = false;
        this.isListening = false;
        this.speechRecognition = null;
        this.translationTimer = null;
        this.translationRequestId = 0;
        this.suggestionRequestId = 0;
        this.fpsCounter = 0;
        this.performanceData = {};
        this.gestureThumbnails = [];

        // Client-side metrics tracking
        this._predictionCount = 0;
        this._predictionStartTime = performance.now();
        this._lastLatency = 0;
        this._totalCorrectPredictions = 0;
        this._totalPredictions = 0;

        // Initialize the application
        this.init();
    }

    /**
     * Initialize the application
     */
    init() {
        console.log('🚀 Initializing ISL Gesture Recognition System...');

        // Bind event listeners
        this.bindEventListeners();

        // Initialize client-side camera
        this.initCamera();

        // Update performance metrics periodically
        setInterval(() => this.updatePerformanceMetrics(), 1000);

        // Initialize UI components
        this.initializeUI();

        console.log('✅ Application initialized successfully');
    }

    /**
     * Bind event listeners to DOM elements
     */
    bindEventListeners() {
        // Speak button
        document.getElementById('speak-btn').addEventListener('click', () => {
            this.speakText();
        });

        document.getElementById('listen-btn').addEventListener('click', () => {
            this.toggleSpeechRecognition();
        });

        document.getElementById('voice-language').addEventListener('change', () => {
            this.refreshTranslatedText(this.currentSentence);
        });

        // Clear button
        document.getElementById('clear-btn').addEventListener('click', () => {
            this.clearSentence();
        });

        // Backspace button
        document.getElementById('backspace-btn')?.addEventListener('click', () => {
            this.backspaceLastCharacter();
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (event) => {
            this.handleKeyboardShortcuts(event);
        });
    }

    /**
     * Initialize client-side camera via SignoraCamera
     */
    async initCamera() {
        if (typeof SignoraCamera === 'undefined') {
            console.error('camera.js not loaded');
            return;
        }

        await this.initializeModelSelector();
        if (!this.selectedModelId) {
            const cameraStatus = document.getElementById('camera-status');
            if (cameraStatus) cameraStatus.textContent = 'No compatible recognition model is loaded';
            return;
        }
        this.camera = new SignoraCamera('#video-stream', {
            fps: this.getPredictionFps(this.selectedModelId),
            buildSentence: true,
            modelId: this.selectedModelId || 'cnn',
            onPrediction: (data) => this.handlePrediction(data),
            onCameraReady: () => {
                const status = document.getElementById('camera-status');
                if (status) status.innerHTML = '<i class="fas fa-circle text-success" style="font-size:0.6em;"></i> Camera Active';
            },
            onError: (e) => {
                const status = document.getElementById('camera-status');
                if (status) status.innerHTML = '<i class="fas fa-circle text-danger" style="font-size:0.6em;"></i> Camera Error';
                console.error('Camera error:', e);
            }
        });

        // Build multi-camera dropdown
        await this.camera.buildCameraSelector('#camera-selector');

        // Start camera and prediction
        const started = await this.camera.start();
        if (started) {
            this.camera.startPredicting();
        }
    }

    getPredictionFps(modelId) {
        if (modelId === 'swatarnik_bigru') return 8;
        if (modelId === 'emergency_lstm') return 12;
        return 3;
    }

    async initializeModelSelector() {
        const select = document.getElementById('recognition-model-select');
        const status = document.getElementById('recognition-model-status');
        if (!select) return;

        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Could not load recognition models.');
            const emergencyStatus = (data.model_status || [])
                .find(model => model.id === 'emergency_lstm');
            const emergencyMessage = document.getElementById('emergency-recognition-status');
            if (emergencyStatus && !emergencyStatus.selectable && emergencyMessage) {
                const reason = emergencyStatus.reason || 'model dependencies or files are unavailable';
                emergencyMessage.textContent = `Emergency Signs model unavailable: ${reason}`;
                emergencyMessage.classList.add('text-warning');
            }

            select.replaceChildren();
            const selectableModels = (data.models || []).filter(model => model.selectable);
            selectableModels.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.name;
                select.appendChild(option);
            });

            if (!selectableModels.length) {
                select.disabled = true;
                status.textContent = 'No compatible models loaded';
                return;
            }

            let savedModelId = null;
            try {
                savedModelId = localStorage.getItem('signora-recognition-model');
            } catch (storageError) {
                console.warn('Model preference could not be read:', storageError);
            }
            this.selectedModelId = selectableModels.some(model => model.id === savedModelId)
                ? savedModelId
                : (data.default_model || selectableModels[0].id);
            select.value = this.selectedModelId;
            select.disabled = false;
            status.textContent = `Using: ${select.options[select.selectedIndex].textContent}`;
            select.addEventListener('change', () => {
                this.selectedModelId = select.value;
                try {
                    localStorage.setItem('signora-recognition-model', this.selectedModelId);
                } catch (storageError) {
                    console.warn('Model preference could not be saved:', storageError);
                }
                if (this.camera) {
                    this.camera.modelId = this.selectedModelId;
                    this.camera.setPredictionFps(this.getPredictionFps(this.selectedModelId));
                }
                status.textContent = `Using: ${select.options[select.selectedIndex].textContent}`;
            });
        } catch (error) {
            select.replaceChildren();
            select.disabled = true;
            status.textContent = 'Model list unavailable';
            console.error('Could not initialize model selector:', error);
        }
    }

    /**
     * Handle prediction results from camera
     */
    handlePrediction(data) {
        if (!data) return;
        const modelStatus = document.getElementById('recognition-model-status');
        const emergencyStatus = document.getElementById('emergency-recognition-status');
        if (data.error) {
            if (modelStatus) modelStatus.textContent = data.error;
            const candidateLabel = document.getElementById('current-character-label');
            if (candidateLabel) candidateLabel.textContent = 'Detected';
            this.currentGesture = '';
            this.currentConfidence = 0.0;
            this.handCount = 0;
            this.updateCurrentCharacter('--');
            this.updateConfidence(0);
            console.error('Gesture prediction failed:', data.error);
            return;
        }
        if (data.emergency_detected && emergencyStatus) {
            const sos = data.emergency_sos || {};
            emergencyStatus.textContent = sos.success
                ? `Emergency sign ${data.gesture} detected. SOS sent.`
                : `Emergency sign ${data.gesture} detected. SOS not sent: ${sos.error || 'unknown error'}`;
            emergencyStatus.classList.toggle('text-danger', !sos.success);
            emergencyStatus.classList.toggle('text-success', Boolean(sos.success));
        }
        if (data.model_buffering) {
            const progress = data.sequence_length
                ? ` ${data.buffered_frames || 0}/${data.sequence_length}`
                : '';
            if (modelStatus) modelStatus.textContent = `Collecting video frames${progress}...`;
            const candidateLabel = document.getElementById('current-character-label');
            if (candidateLabel) candidateLabel.textContent = 'Detected';
            this.currentGesture = '';
            this.currentConfidence = 0.0;
            this.handCount = data.hand_count || 0;
            this.updateCurrentCharacter('--');
            this.updateConfidence(0);
            return;
        }
        const candidateLabel = document.getElementById('current-character-label');
        const displayedGesture = data.stable
            ? data.gesture
            : (data.candidate_gesture || '');
        if (candidateLabel) {
            candidateLabel.textContent = data.stable ? 'Detected' : 'Candidate';
        }
        const modelSelect = document.getElementById('recognition-model-select');
        if (modelStatus && modelSelect) {
            const candidateStatus = displayedGesture && !data.stable
                ? ` · ${displayedGesture} is not confirmed`
                : '';
            modelStatus.textContent = `Using: ${modelSelect.options[modelSelect.selectedIndex].textContent}${candidateStatus}`;
        }

        // Track client-side metrics
        this._predictionCount++;
        this._totalPredictions++;
        if (data.confidence && data.confidence > 0.6) this._totalCorrectPredictions++;
        if (data._latency) this._lastLatency = data._latency;

        // Update gesture display
        if (displayedGesture !== this.currentGesture) {
            this.currentGesture = displayedGesture;
            this.updateCurrentCharacter(displayedGesture || '--');
        }

        // Update confidence
        const confidence = data.stable
            ? data.confidence
            : (displayedGesture ? (data.candidate_confidence || 0) : 0);
        if (confidence !== this.currentConfidence) {
            this.currentConfidence = confidence;
            this.updateConfidence(confidence);
        }

        // Update hand count
        if (data.hand_count !== undefined && data.hand_count !== this.handCount) {
            this.handCount = data.hand_count;
        }

        // Update sentence if server built one
        if (data.sentence !== undefined && data.sentence !== this.currentSentence) {
            this.currentSentence = data.sentence;
            this.updateSentenceDisplay(data.sentence);
            this.updateWordSuggestions(data.sentence);
        }
    }

    /**
     * Initialize UI components
     */
    initializeUI() {
        this.initializeSpeechRecognition();

        // Add loading states
        this.showLoadingState('Initializing camera...');

        // Hide loading after 3 seconds
        setTimeout(() => {
            this.hideLoadingState();
        }, 3000);

        // Initialize tooltips
        this.initializeTooltips();

    }

    /**
     * Update gesture recognition data
     */
    async updateGestureData() {
        if (this.isProcessing) return;

        try {
            this.isProcessing = true;
            const response = await fetch('/api/gesture_data');
            const data = await response.json();

            if (data.error) {
                console.error('Error fetching gesture data:', data.error);
                return;
            }

            // Update current sentence if changed (backend should send accumulated text)
            if (data.current_sentence !== this.currentSentence) {
                this.currentSentence = data.current_sentence;
                this.updateSentenceDisplay(data.current_sentence);
                this.updateWordSuggestions(data.current_sentence);
                console.log('Sentence updated:', data.current_sentence); // Logging for debugging

                // Removed automatic speaking - now only speaks when speak button is pressed
            }

            // Update current gesture and confidence
            if (data.current_gesture !== this.currentGesture) {
                this.currentGesture = data.current_gesture;
                this.updateCurrentCharacter(this.currentGesture || '--');
            }

            if (data.confidence !== this.currentConfidence) {
                this.currentConfidence = data.confidence;
                this.updateConfidence(data.confidence);
            }

            // Update hand count
            if (data.hand_count !== this.handCount) {
                this.handCount = data.hand_count;
                this.updateHandCountDisplay(data.hand_count);
            }

        } catch (error) {
            console.error('Error updating gesture data:', error);
        } finally {
            this.isProcessing = false;
        }
    }

    /**
     * Update performance metrics (client-side computed)
     */
    updatePerformanceMetrics() {
        const now = performance.now();
        const elapsed = (now - this._predictionStartTime) / 1000;
        const fps = elapsed > 0 ? Math.round(this._predictionCount / elapsed) : 0;
        const accuracy = this._totalPredictions > 0
            ? this._totalCorrectPredictions / this._totalPredictions : 0;
        const latency = this._lastLatency || 0;

        // Reset FPS counter every 3 seconds
        if (elapsed > 3) {
            this._predictionCount = 0;
            this._predictionStartTime = now;
        }

        this.displayPerformanceMetrics({
            avg_fps: fps,
            accuracy: accuracy,
            latency: latency
        });
    }

    /**
     * Update FPS counter
     */
    updateFPSCounter() {
        const fpsElement = document.getElementById('fps-counter');
        if (this.performanceData.fps) {
            fpsElement.textContent = Math.round(this.performanceData.fps);
            fpsElement.className = this.performanceData.fps > 25 ? 'text-success' : 'text-warning';
        }
    }

    /**
     * Update sentence display
     */
    updateSentenceDisplay(sentence) {
        const display = document.getElementById('sentence-display');
        display.value = sentence || '';
        this.refreshTranslatedText(sentence || '');

        // Add animation
        display.classList.add('slide-up');
        setTimeout(() => {
            display.classList.remove('slide-up');
        }, 300);

        // Extract and display current character from last part of sentence
        if (sentence && sentence.length > 0) {
            const parts = sentence.trim().split(/\s+/);
            const lastPart = parts[parts.length - 1];
            if (lastPart.length > 0) {
                this.updateCurrentCharacter(lastPart.charAt(lastPart.length - 1).toUpperCase());
            }
        } else {
            this.updateCurrentCharacter('--');
        }
    }

    refreshTranslatedText(sentence) {
        const translatedDisplay = document.getElementById('translated-display');
        const languageSelect = document.getElementById('voice-language');
        const language = languageSelect.value;
        const requestId = ++this.translationRequestId;

        window.clearTimeout(this.translationTimer);
        if (!sentence.trim()) {
            translatedDisplay.value = '';
            return;
        }
        if (language === 'en') {
            translatedDisplay.value = sentence;
            return;
        }

        translatedDisplay.value = 'Translating...';
        this.translationTimer = window.setTimeout(async () => {
            try {
                const response = await fetch('/api/translate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: sentence.trim(), lang: language })
                });
                const result = await response.json();
                if (!response.ok || !result.success) {
                    throw new Error(result.error || 'Translation failed.');
                }
                if (requestId === this.translationRequestId) {
                    translatedDisplay.value = result.translated;
                }
            } catch (error) {
                if (requestId === this.translationRequestId) {
                    translatedDisplay.value = `Translation unavailable: ${error.message}`;
                }
            }
        }, 350);
    }



    /**
     * Update current character display
     */
    updateCurrentCharacter(character) {
        const element = document.getElementById('current-character');
        element.textContent = character;
        element.classList.add('pulse');

        setTimeout(() => {
            element.classList.remove('pulse');
        }, 500);
    }

    /**
     * Update hand count display
     */
    updateHandCountDisplay(count) {
        // Update any hand count displays if needed
        console.log(`Hand count: ${count}`);
    }

    /**
     * Update confidence bar
     */
    updateConfidence(confidence) {
        const confidenceBar = document.getElementById('confidence-bar');
        const confidenceText = document.getElementById('confidence-text');

        const percentage = Math.round(confidence * 100);
        confidenceBar.style.width = percentage + '%';
        confidenceText.textContent = percentage + '%';

        // Change color based on confidence level
        confidenceBar.className = 'progress-bar ' +
            (percentage >= 80 ? 'bg-success' :
                percentage >= 60 ? 'bg-warning' : 'bg-danger');

        // Add animation for high confidence
        if (percentage >= 80) {
            confidenceBar.classList.add('pulse');
            setTimeout(() => {
                confidenceBar.classList.remove('pulse');
            }, 1000);
        }
    }

    /**
     * Update word suggestions via backend API
     */
    async updateWordSuggestions(sentence) {
        const suggestionsContainer = document.getElementById('suggestions');
        const requestId = ++this.suggestionRequestId;
        suggestionsContainer.innerHTML = '<div class="text-muted small">Loading suggestions...</div>'; // Loading state

        try {
            const response = await fetch('/api/suggestions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: sentence || '' })
            });

            const data = await response.json();
            if (requestId !== this.suggestionRequestId) return;

            if (data.error) {
                console.error('Error fetching suggestions:', data.error);
                suggestionsContainer.innerHTML = '<div class="text-muted small">Suggestions unavailable</div>';
                return;
            }

            suggestionsContainer.innerHTML = '';
            if (data.suggestions && data.suggestions.length > 0) {
                data.suggestions.forEach(word => {
                    const button = document.createElement('button');
                    button.className = 'suggestion-btn fw-bold';
                    button.textContent = word;
                    button.onclick = () => this.applySuggestion(word);
                    suggestionsContainer.appendChild(button);
                });
            } else {
                suggestionsContainer.innerHTML = '<div class="text-muted small">No suggestions</div>';
            }
        } catch (error) {
            if (requestId !== this.suggestionRequestId) return;
            console.error('Error updating word suggestions:', error);
            suggestionsContainer.innerHTML = '<div class="text-muted small">Error loading suggestions</div>';
        }
    }

    /**
     * Apply word suggestion
     */
    async applySuggestion(word) {
        const words = this.currentSentence.split(' ');
        if (words.length > 0) {
            words[words.length - 1] = word;
        } else {
            words.push(word);
        }
        const newSentence = words.join(' ') + ' ';

        this.currentSentence = newSentence;
        this.updateSentenceDisplay(newSentence);
        this.updateWordSuggestions(newSentence);

        await this.sendSentenceUpdate(newSentence);
    }

    /**
     * Display performance metrics with color coding
     */
    displayPerformanceMetrics(data) {
        // Update average FPS with color class
        const avgFpsElement = document.getElementById('avg-fps');
        if (data.avg_fps !== undefined) {
            const fpsValue = Math.round(data.avg_fps);
            avgFpsElement.textContent = fpsValue;
            avgFpsElement.className = 'metric-value ' +
                (fpsValue > 25 ? 'good' : fpsValue > 15 ? 'warning' : 'danger');
        }

        // Update accuracy with color class
        const accuracyElement = document.getElementById('accuracy');
        if (data.accuracy !== undefined) {
            const accValue = Math.round(data.accuracy * 100);
            accuracyElement.textContent = accValue + '%';
            accuracyElement.className = 'metric-value ' +
                (accValue >= 80 ? 'good' : accValue >= 60 ? 'warning' : 'danger');
        }

        // Update latency with color class (lower is better)
        const latencyElement = document.getElementById('latency');
        if (data.latency !== undefined) {
            const latValue = Math.round(data.latency);
            latencyElement.textContent = latValue + 'ms';
            latencyElement.className = 'metric-value ' +
                (latValue < 100 ? 'good' : latValue < 200 ? 'warning' : 'danger');
        }
    }

    /**
     * Speak current text
     */
    async speakText() {
        const speakBtn = document.getElementById('speak-btn');
        const originalText = speakBtn.innerHTML;
        const status = document.getElementById('voice-status');

        if (this.isSpeaking || !this.currentSentence || !this.currentSentence.trim()) {
            if (!this.currentSentence || !this.currentSentence.trim()) {
                status.textContent = 'There is no text to speak yet.';
            }
            return;
        }

        this.isSpeaking = true;

        try {
            speakBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Speaking...';
            speakBtn.disabled = true;

            const languageSelect = document.getElementById('voice-language');
            const language = languageSelect.value;
            const locale = languageSelect.selectedOptions[0].dataset.locale;
            let text = this.currentSentence.trim();

            if (language !== 'en') {
                status.textContent = 'Translating text...';
                const response = await fetch('/api/translate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, lang: language })
                });
                const result = await response.json();
                if (!response.ok || !result.success) {
                    throw new Error(result.error || 'Translation failed.');
                }
                text = result.translated;
            }

            const voices = window.speechSynthesis
                ? await this.getAvailableSpeechVoices()
                : [];
            const normalizeLocale = value => value.replace('_', '-').toLowerCase();
            const matchingVoice = voices.find(voice => normalizeLocale(voice.lang) === normalizeLocale(locale))
                || voices.find(voice => normalizeLocale(voice.lang).startsWith(`${language}-`));
            let browserSpeechError = null;

            if (matchingVoice) {
                try {
                    await new Promise((resolve, reject) => {
                        const utterance = new SpeechSynthesisUtterance(text);
                        utterance.lang = locale;
                        utterance.onend = resolve;
                        utterance.onerror = event => reject(new Error(`Speech output failed: ${event.error}`));
                        utterance.voice = matchingVoice;

                        window.speechSynthesis.cancel();
                        window.speechSynthesis.speak(utterance);
                    });
                } catch (error) {
                    browserSpeechError = error;
                }
            }

            if (!matchingVoice || browserSpeechError) {
                status.textContent = 'Using online speech for this language...';
                await this.playGeneratedSpeech(text, language);
                status.textContent = `Finished speaking in ${languageSelect.selectedOptions[0].text}.`;
            } else {
                status.textContent = `Finished speaking in ${languageSelect.selectedOptions[0].text}.`;
            }
            speakBtn.innerHTML = '<i class="fas fa-check me-2"></i>Spoken!';

        } catch (error) {
            console.error('Error speaking text:', error);
            status.textContent = error.message;
            speakBtn.innerHTML = originalText;
        } finally {
            setTimeout(() => {
                speakBtn.innerHTML = originalText;
            }, 2000);
            speakBtn.disabled = false;
            this.isSpeaking = false;
        }
    }

    async playGeneratedSpeech(text, language) {
        const response = await fetch('/api/speech-audio', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, lang: language })
        });
        if (!response.ok) {
            let message = 'Could not generate speech audio.';
            try {
                const result = await response.json();
                message = result.error || message;
            } catch (error) {
                console.warn('Speech audio endpoint returned a non-JSON error response.', error);
            }
            throw new Error(message);
        }

        const audioUrl = URL.createObjectURL(await response.blob());
        try {
            const audio = new Audio(audioUrl);
            await new Promise((resolve, reject) => {
                audio.onended = resolve;
                audio.onerror = () => reject(new Error('Generated speech audio could not be played.'));
                audio.play().catch(reject);
            });
        } finally {
            URL.revokeObjectURL(audioUrl);
        }
    }

    async getAvailableSpeechVoices() {
        const speechSynthesis = window.speechSynthesis;
        const voices = speechSynthesis.getVoices();
        if (voices.length) return voices;

        return new Promise(resolve => {
            const timeout = window.setTimeout(() => {
                speechSynthesis.removeEventListener('voiceschanged', handleVoicesChanged);
                resolve(speechSynthesis.getVoices());
            }, 1500);
            const handleVoicesChanged = () => {
                window.clearTimeout(timeout);
                resolve(speechSynthesis.getVoices());
            };
            speechSynthesis.addEventListener('voiceschanged', handleVoicesChanged, { once: true });
        });
    }

    initializeSpeechRecognition() {
        const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const listenBtn = document.getElementById('listen-btn');
        const status = document.getElementById('voice-status');

        if (!Recognition) {
            listenBtn.disabled = true;
            status.textContent = 'Speech input is not supported by this browser. You can still use voice output.';
            return;
        }

        this.speechRecognition = new Recognition();
        this.speechRecognition.continuous = false;
        this.speechRecognition.interimResults = false;
        this.speechRecognition.onresult = async (event) => {
            const transcript = event.results[0][0].transcript.trim();
            if (!transcript) {
                status.textContent = 'No speech was recognized. Please try again.';
                return;
            }

            const separator = this.currentSentence && !/\s$/.test(this.currentSentence) ? ' ' : '';
            this.currentSentence = `${this.currentSentence}${separator}${transcript}`;
            this.updateSentenceDisplay(this.currentSentence);
            this.updateWordSuggestions(this.currentSentence);
            const saved = await this.sendSentenceUpdate(this.currentSentence);
            if (saved) {
                status.textContent = `Added recognized speech: "${transcript}"`;
            } else {
                status.textContent = 'Speech was recognized but could not be saved to the transcript.';
            }
        };
        this.speechRecognition.onerror = (event) => {
            const messages = {
                'not-allowed': 'Microphone access was denied. Allow microphone access and try again.',
                'service-not-allowed': 'Speech recognition is unavailable for this site.',
                'no-speech': 'No speech was detected. Please try again.'
            };
            status.textContent = messages[event.error] || `Speech recognition failed: ${event.error}.`;
        };
        this.speechRecognition.onend = () => {
            this.isListening = false;
            listenBtn.innerHTML = '<i class="fas fa-microphone me-2"></i>Speak to Text';
            listenBtn.classList.remove('btn-danger');
            listenBtn.classList.add('btn-outline-primary');
        };
    }

    toggleSpeechRecognition() {
        const recognition = this.speechRecognition;
        const listenBtn = document.getElementById('listen-btn');
        const languageSelect = document.getElementById('voice-language');
        const status = document.getElementById('voice-status');

        if (!recognition) return;
        if (this.isListening) {
            recognition.stop();
            status.textContent = 'Stopped listening.';
            return;
        }

        recognition.lang = languageSelect.selectedOptions[0].dataset.locale;
        try {
            recognition.start();
            this.isListening = true;
            listenBtn.innerHTML = '<i class="fas fa-stop me-2"></i>Stop Listening';
            listenBtn.classList.remove('btn-outline-primary');
            listenBtn.classList.add('btn-danger');
            status.textContent = `Listening in ${languageSelect.selectedOptions[0].text}...`;
        } catch (error) {
            status.textContent = `Could not start speech recognition: ${error.message}`;
        }
    }



    /**
     * Clear current sentence
     */
    async clearSentence() {
        const resumePredictions = this.camera?.predicting === true;
        if (resumePredictions) this.camera.stopPredicting();

        try {
            await this.camera?.waitForPredictionIdle();
            const response = await fetch('/api/clear_sentence', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            if (!response.ok) {
                throw new Error(`Clear request failed (${response.status}).`);
            }
            const result = await response.json();

            if (result.success) {
                this.currentSentence = '';
                this.updateSentenceDisplay('');
                this.updateCurrentCharacter('--');
                this.updateConfidence(0);
                this.updateWordSuggestions('');
                document.getElementById('voice-status').textContent = 'Transcript cleared.';
            } else {
                throw new Error(result.error || 'Could not clear the transcript.');
            }
        } catch (error) {
            console.error('Error clearing sentence:', error);
            document.getElementById('voice-status').textContent = error.message;
        } finally {
            if (resumePredictions) this.camera.startPredicting();
        }
    }

    /**
     * Handle video stream errors
     */
    handleVideoError() {
        const statusElement = document.getElementById('camera-status');
        statusElement.innerHTML = '<i class="fas fa-circle text-danger"></i> Camera Error';
        console.error('Video stream error');
    }

    /**
     * Handle video stream load
     */
    handleVideoLoad() {
        const statusElement = document.getElementById('camera-status');
        statusElement.innerHTML = '<i class="fas fa-circle text-success"></i> Camera Active';
    }

    /**
     * Handle keyboard shortcuts
     */
    handleKeyboardShortcuts(event) {
        // Ctrl/Cmd + Space: Speak
        if ((event.ctrlKey || event.metaKey) && event.code === 'Space') {
            event.preventDefault();
            this.speakText();
        }

        // Ctrl/Cmd + Backspace: Clear sentence
        if ((event.ctrlKey || event.metaKey) && event.code === 'Backspace') {
            event.preventDefault();
            this.clearSentence();
        }

        // Backspace: Remove last character
        if (event.code === 'Backspace' && !event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey) {
            event.preventDefault();
            this.backspaceLastCharacter();
        }

        // Plus key (+): Add space
        if (event.code === 'Equal' && event.shiftKey && !event.ctrlKey && !event.metaKey && !event.altKey) {
            event.preventDefault();
            this.addSpace();
        }
    }

    /**
     * Show loading state
     */
    showLoadingState(message = 'Loading...') {
        // Implementation for loading state
        console.log('Loading:', message);
    }

    /**
     * Hide loading state
     */
    hideLoadingState() {
        // Implementation for hiding loading state
        console.log('Loading complete');
    }

    /**
     * Initialize tooltips
     */
    initializeTooltips() {
        // Add tooltips to buttons
        document.getElementById('speak-btn').title = 'Translate and speak text in the selected language (Ctrl+Space)';
        document.getElementById('clear-btn').title = 'Clear text (Ctrl+Backspace)';
    }

    /**
     * Send sentence update to backend
     */
    async sendSentenceUpdate(sentence) {
        const resumePredictions = this.camera?.predicting === true;
        if (resumePredictions) this.camera.stopPredicting();

        try {
            await this.camera?.waitForPredictionIdle();
            const response = await fetch('/api/update_sentence', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ sentence: sentence })
            });
            if (!response.ok) {
                console.error(`Sentence update failed with HTTP ${response.status}`);
                document.getElementById('voice-status').textContent =
                    `Transcript could not be synchronized (HTTP ${response.status}).`;
                return false;
            }
            const result = await response.json();
            if (!result.success) {
                document.getElementById('voice-status').textContent =
                    result.error || 'Transcript could not be synchronized.';
                return false;
            }

            this.currentSentence = sentence;
            this.updateSentenceDisplay(sentence);
            this.updateWordSuggestions(sentence);
            return true;
        } catch (error) {
            console.error('Error sending sentence update:', error);
            document.getElementById('voice-status').textContent =
                'Transcript could not be synchronized. Please try again.';
            return false;
        } finally {
            if (resumePredictions) this.camera.startPredicting();
        }
    }

    /**
     * Remove last character from sentence
     */
    async backspaceLastCharacter() {
        if (!this.currentSentence || this.currentSentence.length === 0) {
            console.log('No characters to remove');
            return;
        }

        // Remove the last character and update local state
        this.currentSentence = this.currentSentence.slice(0, -1);

        // Update the display
        this.updateSentenceDisplay(this.currentSentence);

        // Send to backend
        await this.sendSentenceUpdate(this.currentSentence);

        console.log('Backspace: Removed last character');
    }

    /**
     * Add space to sentence
     */
    async addSpace() {
        // Add space to the current sentence
        const newSentence = this.currentSentence + ' ';

        this.currentSentence = newSentence;
        // Update the display
        this.updateSentenceDisplay(newSentence);

        // Send to backend
        await this.sendSentenceUpdate(newSentence);

        console.log('Space: Added space to sentence');
    }

    /**
     * Check application health
     */
    async checkHealth() {
        try {
            const response = await fetch('/health');
            const health = await response.json();
            console.log('Application health:', health);
            return health;
        } catch (error) {
            console.error('Health check failed:', error);
            return null;
        }
    }

}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.islApp = new ISLGestureApp();

    // Perform initial health check
    window.islApp.checkHealth().then(health => {
        if (health && health.status === 'healthy') {
            console.log('✅ Application is healthy and ready!');
        } else {
            console.warn('⚠️ Application health check failed');
        }
    });
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        console.log('Page became visible - resuming updates');
        // Resume updates if needed
    } else {
        console.log('Page became hidden - pausing updates');
        // Pause updates to save resources
    }
});

// Handle connection errors
window.addEventListener('online', () => {
    console.log('Connection restored');
});

window.addEventListener('offline', () => {
    console.log('Connection lost');
});
