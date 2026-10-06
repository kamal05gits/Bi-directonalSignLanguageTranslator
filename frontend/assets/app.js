const $ = (id) => document.getElementById(id);
const els = Object.fromEntries(['apiStatus','video','canvas','landmarkOverlay','cameraEmpty','cameraButton','pauseButton','captureButton','flipButton','stageTitle','guideLabel','privacyNote','predictionEmpty','predictionResult','predictionHint','letter','confidence','confidenceMeter','predictionState','modelChips','addButton','alternatives','message','characterCount','clearButton','spaceButton','periodButton','backspaceButton','speakButton','replayButton','stopSpeechButton','muteToggle','fingerSequence','fingerNote','toast','countdown','pausedOverlay','languageSelect','translateButton','speakTranslatedButton','translationOutput','emergencyButton','emergencyPanel','emergencyClose','emergencyNote','emergencyStatus','emergencyPhrases','emergencyBanner'].map(id => [id, $(id)]));
const modeButtons = [...document.querySelectorAll('.model-option')];
const modelDots = Object.fromEntries([...document.querySelectorAll('[data-model-dot]')].map(dot => [dot.dataset.modelDot, dot]));

const WORD_SECONDS = 3;
const HANDS_VERSION = '0.4.1675469240';
const HANDS_CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${HANDS_VERSION}`;
const DRAWING_VERSION = '0.3.1675466124';
const DRAWING_CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@${DRAWING_VERSION}`;
const CONTINUOUS_INTERVAL_MS = 350;
const NO_HAND_FRAMES_BEFORE_SPACE = 4;
const VOICE_LANGUAGE_TAGS = {ta: 'ta-IN', hi: 'hi-IN'};

let stream = null;
let facingMode = 'user';
let mode = 'combined';
let modelHealth = {};
let currentLabel = '';
let toastTimer;
let handsPromise = null;
let handsInstance = null;
let drawingUtilsPromise = null;
let recording = false;
let paused = false;
let lastUtteranceText = '';
let translatedText = '';
let translatedLanguage = '';

// --- Continuous (live stream) recognition state ---
let continuousSessionId = null;
let continuousTimer = null;
let continuousNoHandStreak = 0;
let continuousBusy = false;

const MODEL_LABELS = {alphabet: 'Photo', fingerspelling: 'Landmarks', word: 'Video'};

const MODES = {
  combined: {
    stageTitle: 'Sign a letter or word',
    guideLabel: 'Fit your hand (and upper body for words)',
    captureLabel: `Record ${WORD_SECONDS}s & combine`,
    hint: 'Records one clip, then runs the photo, landmark, and video models together and merges their votes into a single prediction.',
    addLabel: 'Add to message',
    addsWord: false,
    privacy: 'One frame, its hand landmarks, and the clip are processed together and never stored.',
  },
  alphabet: {
    stageTitle: 'Show a sign',
    guideLabel: 'Place one hand here',
    captureLabel: 'Recognize sign',
    hint: 'Take a photo of one alphabet sign.',
    addLabel: 'Add letter',
    addsWord: false,
    privacy: 'Images are processed for recognition and are not stored.',
  },
  fingerspelling: {
    stageTitle: 'Show a letter sign',
    guideLabel: 'Keep your whole hand in view',
    captureLabel: 'Recognize sign',
    hint: 'Reads your hand skeleton, so the background does not matter.',
    addLabel: 'Add letter',
    addsWord: false,
    privacy: 'Only landmark coordinates leave your browser — never the video.',
  },
  word: {
    stageTitle: 'Record one word',
    guideLabel: 'Fit your hands and upper body',
    captureLabel: `Record ${WORD_SECONDS}s sign`,
    hint: `Records about ${WORD_SECONDS} seconds of one continuous sign.`,
    addLabel: 'Add word',
    addsWord: true,
    privacy: 'Clips are processed for recognition and are not stored.',
  },
  continuous: {
    stageTitle: 'Sign continuously',
    guideLabel: 'Keep your whole hand in view',
    captureLabel: 'Start continuous recognition',
    hint: 'Hold each letter steady; SignBridge builds the sentence automatically. Pause with no hand visible to insert a space.',
    addLabel: 'Add letter',
    addsWord: false,
    privacy: 'Only landmark coordinates leave your browser, streamed continuously while active — never the video.',
  },
};

function toast(message, error = false) {
  els.toast.textContent = message;
  els.toast.className = `toast show${error ? ' error' : ''}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => els.toast.className = 'toast', 3200);
}

async function checkApi() {
  try {
    const response = await fetch('/api/health');
    if (!response.ok) throw new Error();
    const data = await response.json();
    els.apiStatus.className = data.model_available ? 'status online' : 'status offline';
    els.apiStatus.querySelector('span').textContent = data.model_available ? 'Service ready' : 'Model missing';
    modelHealth = data.models || {};
    // Continuous mode reuses the fingerspelling model under the hood.
    modelHealth.continuous = modelHealth.fingerspelling;
    // Combined mode runs every model that is available on one capture.
    const availableModels = ['alphabet', 'fingerspelling', 'word'].filter(name => modelHealth[name] && modelHealth[name].available);
    modelHealth.combined = {
      available: availableModels.length > 0,
      detail: availableModels.length === 3
        ? 'Runs all three models and merges their votes'
        : `Runs ${availableModels.join(' + ') || 'no models'} — deploy the missing models for a full ensemble`,
    };
    MODES.combined.captureLabel = modelHealth.word && modelHealth.word.available
      ? `Record ${WORD_SECONDS}s & combine`
      : 'Capture & combine models';
    for (const [name, dot] of Object.entries(modelDots)) {
      const health = modelHealth[name];
      const ok = Boolean(health && health.available);
      dot.className = `model-dot ${ok ? 'ok' : 'down'}`;
      dot.title = health ? health.detail : 'Unknown';
    }
    if (mode === 'combined' && !recording) els.captureButton.textContent = MODES.combined.captureLabel;
    if (modelHealth.word && !modelHealth.word.available && mode === 'word') toast(modelHealth.word.detail, true);
  } catch {
    els.apiStatus.className = 'status offline';
    els.apiStatus.querySelector('span').textContent = 'Service unavailable';
  }
}

function setMode(nextMode) {
  if (recording) return toast('Stop the recording before switching models.', true);
  if (mode === 'continuous' && continuousTimer) stopContinuous();
  mode = nextMode;
  const config = MODES[mode];
  modeButtons.forEach(button => {
    const active = button.dataset.mode === mode;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
  });
  els.stageTitle.textContent = config.stageTitle;
  els.guideLabel.textContent = config.guideLabel;
  els.captureButton.textContent = config.captureLabel;
  els.predictionHint.textContent = config.hint;
  els.privacyNote.textContent = config.privacy;
  els.addButton.textContent = config.addLabel;
  els.addButton.hidden = mode === 'continuous';
  els.message.readOnly = mode === 'continuous';
  els.modelChips.hidden = true;
  clearLandmarkOverlay();
  if (modelHealth[mode] && !modelHealth[mode].available) toast(modelHealth[mode].detail, true);
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) return toast('Camera access is not supported in this browser.', true);
  stopCamera();
  try {
    stream = await navigator.mediaDevices.getUserMedia({video: {facingMode, width: {ideal: 960}, height: {ideal: 720}}, audio: false});
    els.video.srcObject = stream;
    await els.video.play();
    els.cameraEmpty.hidden = true;
    els.captureButton.disabled = false;
    els.pauseButton.disabled = false;
    els.cameraButton.textContent = 'Stop camera';
    paused = false;
    els.pausedOverlay.hidden = true;
  } catch (error) {
    toast(error.name === 'NotAllowedError' ? 'Camera permission was denied. Allow it in browser settings.' : 'Could not start your camera.', true);
  }
}

function stopCamera() {
  if (continuousTimer) stopContinuous();
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  recording = false;
  paused = false;
  els.video.srcObject = null;
  els.cameraEmpty.hidden = false;
  els.captureButton.disabled = true;
  els.pauseButton.disabled = true;
  els.pausedOverlay.hidden = true;
  els.cameraButton.textContent = 'Start camera';
  els.countdown.hidden = true;
  clearLandmarkOverlay();
}

function togglePause() {
  if (!stream) return;
  paused = !paused;
  stream.getVideoTracks().forEach(track => { track.enabled = !paused; });
  els.pausedOverlay.hidden = !paused;
  els.pauseButton.textContent = paused ? '▶' : '⏸';
  els.pauseButton.title = paused ? 'Resume camera' : 'Pause camera';
  if (paused && continuousTimer) clearInterval(continuousTimer), continuousTimer = null;
  else if (!paused && mode === 'continuous' && continuousSessionId && !continuousTimer) {
    continuousTimer = setInterval(stepContinuous, CONTINUOUS_INTERVAL_MS);
  }
}

// Draws the current video frame to the hidden canvas. The selfie camera is
// mirrored only for the photo model, matching what the user sees on screen.
// The landmark model instead sees the raw camera orientation, mirroring how
// its training vectors were captured with OpenCV.
function drawFrame(mirror) {
  const size = Math.min(els.video.videoWidth, els.video.videoHeight);
  const sx = (els.video.videoWidth - size) / 2;
  const sy = (els.video.videoHeight - size) / 2;
  els.canvas.width = 640;
  els.canvas.height = 640;
  const context = els.canvas.getContext('2d');
  if (mirror && facingMode === 'user') { context.translate(640, 0); context.scale(-1, 1); }
  context.drawImage(els.video, sx, sy, size, size, 0, 0, 640, 640);
  context.setTransform(1, 0, 0, 1, 0, 0);
  return context;
}

async function captureAlphabet() {
  if (!stream || !els.video.videoWidth) return;
  els.captureButton.disabled = true;
  els.captureButton.textContent = 'Recognizing…';
  drawFrame(true);
  const blob = await new Promise(resolve => els.canvas.toBlob(resolve, 'image/jpeg', .9));
  const form = new FormData();
  form.append('file', blob, 'capture.jpg');
  try {
    const data = await postForm('/api/predict/alphabet', form);
    showPrediction(data);
  } catch (error) {
    toast(error.message, true);
  } finally {
    els.captureButton.disabled = false;
    els.captureButton.textContent = MODES.alphabet.captureLabel;
  }
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.onload = resolve;
    script.onerror = () => reject(new Error('Could not load the hand-tracking library. Check your internet connection and try again.'));
    document.head.appendChild(script);
  });
}

async function ensureHandTracker() {
  if (handsInstance) return handsInstance;
  handsPromise = handsPromise || (async () => {
    if (!window.Hands) await loadScript(`${HANDS_CDN}/hands.js`);
    const hands = new window.Hands({locateFile: file => `${HANDS_CDN}/${file}`});
    hands.setOptions({maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.6, minTrackingConfidence: 0.5});
    handsInstance = hands;
    return hands;
  })();
  try {
    return await handsPromise;
  } catch (error) {
    handsPromise = null;
    throw error;
  }
}

async function ensureDrawingUtils() {
  if (window.drawConnectors && window.drawLandmarks) return;
  drawingUtilsPromise = drawingUtilsPromise || loadScript(`${DRAWING_CDN}/drawing_utils.js`);
  await drawingUtilsPromise;
}

function detectHands(source, hands) {
  return new Promise((resolve, reject) => {
    hands.onResults(resolve);
    hands.send({image: source}).catch(reject);
  });
}

// Visual feedback for the "Hand Detection & Landmark Extraction" module: draw
// the tracked skeleton over the live video so the user can see what the
// model sees, instead of the detector being an invisible black box.
function clearLandmarkOverlay() {
  if (!els.landmarkOverlay) return;
  const context = els.landmarkOverlay.getContext('2d');
  context.clearRect(0, 0, els.landmarkOverlay.width, els.landmarkOverlay.height);
}

async function drawLandmarkOverlay(results) {
  if (!els.landmarkOverlay || !els.video.videoWidth) return;
  els.landmarkOverlay.width = els.video.clientWidth;
  els.landmarkOverlay.height = els.video.clientHeight;
  const context = els.landmarkOverlay.getContext('2d');
  context.clearRect(0, 0, els.landmarkOverlay.width, els.landmarkOverlay.height);
  const hands = results.multiHandLandmarks || [];
  if (!hands.length) return;
  try {
    await ensureDrawingUtils();
  } catch {
    return; // Overlay is cosmetic; recognition still works without it.
  }
  context.save();
  if (facingMode === 'user') { context.translate(els.landmarkOverlay.width, 0); context.scale(-1, 1); }
  hands.forEach(points => {
    window.drawConnectors(context, points, window.HAND_CONNECTIONS, {color: '#c8f169', lineWidth: 3});
    window.drawLandmarks(context, points, {color: '#111827', lineWidth: 1, radius: 3});
  });
  context.restore();
}

// Builds the 126-value vector the fingerspelling model was trained on:
// left hand in slots 0-62, right hand in 63-125, wrist-relative (x, y, z).
// See backend/app/vision/landmarks.py.
function landmarksToVector(results) {
  const vector = new Array(126).fill(0);
  const hands = results.multiHandLandmarks || [];
  const handedness = results.multiHandedness || [];
  hands.forEach((points, index) => {
    const label = handedness[index] && handedness[index].label;
    const slot = label === 'Left' ? 0 : 63;
    const wrist = points[0];
    points.forEach((point, pointIndex) => {
      vector[slot + pointIndex * 3] = point.x - wrist.x;
      vector[slot + pointIndex * 3 + 1] = point.y - wrist.y;
      vector[slot + pointIndex * 3 + 2] = point.z - wrist.z;
    });
  });
  return vector;
}

async function captureFingerspelling() {
  if (!stream || !els.video.videoWidth) return;
  els.captureButton.disabled = true;
  els.captureButton.textContent = handsInstance ? 'Recognizing…' : 'Loading hand tracker…';
  try {
    const hands = await ensureHandTracker();
    els.captureButton.textContent = 'Recognizing…';
    drawFrame(false);
    const results = await detectHands(els.canvas, hands);
    drawLandmarkOverlay(results);
    if (!results.multiHandLandmarks || !results.multiHandLandmarks.length) {
      return toast('No hand detected. Fill the guide with one well-lit hand.', true);
    }
    const data = await postJson('/api/predict/fingerspelling', {landmarks: landmarksToVector(results)});
    showPrediction(data);
  } catch (error) {
    toast(error.message, true);
  } finally {
    els.captureButton.disabled = false;
    els.captureButton.textContent = MODES.fingerspelling.captureLabel;
  }
}

// --- Continuous (live stream) recognition ---------------------------------

async function ensureContinuousSession() {
  if (continuousSessionId) return continuousSessionId;
  const data = await postJson('/api/continuous/session', {});
  continuousSessionId = data.session_id;
  return continuousSessionId;
}

function renderContinuousState(data) {
  els.modelChips.hidden = true;
  els.message.value = data.text || '';
  updateMessage();
  if (data.label) {
    els.predictionEmpty.hidden = true;
    els.predictionResult.hidden = false;
    els.letter.textContent = data.label;
    const percent = Math.round((data.confidence || 0) * 100);
    els.confidence.textContent = `${percent}% confidence`;
    els.confidenceMeter.style.width = `${percent}%`;
    els.predictionState.textContent = data.accepted ? 'Added to sentence' : (data.reason || 'Stabilizing…');
  } else {
    els.predictionState.textContent = data.reason || 'Show a hand to begin';
  }
}

async function stepContinuous() {
  if (!stream || paused || continuousBusy || !els.video.videoWidth) return;
  continuousBusy = true;
  try {
    const hands = await ensureHandTracker();
    drawFrame(false);
    const results = await detectHands(els.canvas, hands);
    drawLandmarkOverlay(results);
    const hasHand = results.multiHandLandmarks && results.multiHandLandmarks.length;
    const vector = hasHand ? landmarksToVector(results) : new Array(126).fill(0);
    const data = await postJson(`/api/continuous/session/${continuousSessionId}/frame`, {landmarks: vector});
    renderContinuousState(data);
    if (data.reason === 'No hand detected') {
      continuousNoHandStreak += 1;
      if (continuousNoHandStreak === NO_HAND_FRAMES_BEFORE_SPACE) {
        const spaced = await postJson(`/api/continuous/session/${continuousSessionId}/space`, {});
        renderContinuousState(spaced);
      }
    } else {
      continuousNoHandStreak = 0;
    }
  } catch (error) {
    toast(error.message, true);
  } finally {
    continuousBusy = false;
  }
}

async function startContinuous() {
  if (!stream) return toast('Start the camera first.', true);
  try {
    els.captureButton.disabled = true;
    els.captureButton.textContent = 'Starting…';
    await ensureHandTracker();
    await ensureContinuousSession();
    continuousNoHandStreak = 0;
    els.predictionEmpty.hidden = true;
    els.predictionResult.hidden = false;
    els.letter.textContent = '—';
    els.predictionState.textContent = 'Show a hand to begin';
    els.confidence.textContent = '0% confidence';
    els.confidenceMeter.style.width = '0%';
    continuousTimer = setInterval(stepContinuous, CONTINUOUS_INTERVAL_MS);
    els.captureButton.textContent = 'Stop continuous recognition';
  } catch (error) {
    toast(error.message, true);
  } finally {
    els.captureButton.disabled = false;
  }
}

function stopContinuous() {
  if (continuousTimer) clearInterval(continuousTimer);
  continuousTimer = null;
  els.captureButton.textContent = MODES.continuous.captureLabel;
}

async function toggleContinuous() {
  if (continuousTimer) stopContinuous();
  else await startContinuous();
}

async function continuousAction(path) {
  if (!continuousSessionId) return false;
  try {
    const data = await postJson(`/api/continuous/session/${continuousSessionId}${path}`, path.includes('punctuation') ? {mark: '.'} : {});
    renderContinuousState(data);
    return true;
  } catch (error) {
    toast(error.message, true);
    return true;
  }
}

function pickVideoMimeType() {
  if (!window.MediaRecorder) return null;
  const candidates = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
  return candidates.find(candidate => MediaRecorder.isTypeSupported(candidate)) || '';
}

// Records a clip of the live stream and resolves with the recorded blob.
// `onTick(remaining)` runs once per countdown second so callers can grab
// frames from mid-sign (used by combined mode to feed the letter models).
async function recordClip(seconds = WORD_SECONDS, onTick = null) {
  const mimeType = pickVideoMimeType();
  if (mimeType === null) throw new Error('Video recording is not supported in this browser.');
  const extension = mimeType.includes('mp4') ? 'mp4' : 'webm';
  let recorder;
  try {
    recorder = new MediaRecorder(stream, mimeType ? {mimeType, videoBitsPerSecond: 2500000} : undefined);
  } catch {
    throw new Error('Could not start video recording in this browser.');
  }
  const chunks = [];
  recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
  const stopped = new Promise(resolve => recorder.onstop = resolve);
  recording = true;
  els.captureButton.disabled = true;
  els.flipButton.disabled = true;
  recorder.start(250);
  for (let remaining = seconds; remaining > 0 && recording; remaining--) {
    els.countdown.textContent = remaining;
    els.countdown.hidden = false;
    els.captureButton.textContent = `Recording… ${remaining}s`;
    if (onTick) await onTick(remaining);
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  els.countdown.hidden = true;
  if (recorder.state !== 'inactive') recorder.stop();
  await stopped;
  recording = false;
  els.flipButton.disabled = false;
  const blob = new Blob(chunks, {type: recorder.mimeType || mimeType || 'video/webm'});
  if (!blob.size) throw new Error('Recording came back empty. Please try again.');
  return {blob, extension};
}

async function recordWord() {
  if (!stream || recording) return;
  els.captureButton.disabled = true;
  try {
    const {blob, extension} = await recordClip(WORD_SECONDS);
    els.captureButton.textContent = 'Recognizing…';
    const form = new FormData();
    form.append('file', blob, `sign.${extension}`);
    const data = await postForm('/api/predict/word', form);
    showPrediction(data);
  } catch (error) {
    toast(error.message, true);
  } finally {
    els.captureButton.disabled = !stream;
    els.captureButton.textContent = MODES.word.captureLabel;
  }
}

// --- Combined mode (all three models, one merged prediction) ---------------

// Grabs one mirrored JPEG for the photo model plus the raw-orientation hand
// landmarks for the fingerspelling model from the same moment.
async function grabFrameAndLandmarks() {
  const hands = await ensureHandTracker();
  drawFrame(false); // raw camera orientation, matching the landmark training data
  const results = await detectHands(els.canvas, hands);
  drawLandmarkOverlay(results);
  const hasHand = results.multiHandLandmarks && results.multiHandLandmarks.length;
  const landmarks = hasHand ? landmarksToVector(results) : null;
  drawFrame(true); // mirrored, matching what the user sees (photo model input)
  const imageBlob = await new Promise(resolve => els.canvas.toBlob(resolve, 'image/jpeg', .9));
  return {imageBlob, landmarks};
}

async function captureCombined() {
  if (!stream || !els.video.videoWidth || recording) return;
  const wordAvailable = modelHealth.word && modelHealth.word.available;
  els.captureButton.disabled = true;
  try {
    let clip = null;
    let frame = null;
    if (wordAvailable) {
      // Load MediaPipe before the countdown so the mid-sign grab never stalls it.
      els.captureButton.textContent = handsInstance ? 'Preparing…' : 'Loading hand tracker…';
      await ensureHandTracker();
      clip = await recordClip(WORD_SECONDS, async remaining => {
        // Grab the frame and landmarks mid-sign (second of three).
        if (remaining === WORD_SECONDS - 1) frame = await grabFrameAndLandmarks();
      });
      if (!frame) frame = await grabFrameAndLandmarks(); // recording was cut short
    } else {
      els.captureButton.textContent = handsInstance ? 'Recognizing…' : 'Loading hand tracker…';
      frame = await grabFrameAndLandmarks();
    }
    els.captureButton.textContent = 'Combining models…';
    if (!frame.landmarks) toast('No hand detected — using the photo model only.', true);
    const form = new FormData();
    form.append('image', frame.imageBlob, 'capture.jpg');
    if (frame.landmarks) form.append('landmarks', JSON.stringify(frame.landmarks));
    if (clip) form.append('video', clip.blob, `sign.${clip.extension}`);
    const data = await postForm('/api/predict/combined', form);
    showCombinedPrediction(data);
  } catch (error) {
    toast(error.message, true);
  } finally {
    els.captureButton.disabled = !stream;
    els.captureButton.textContent = MODES.combined.captureLabel;
  }
}

function showCombinedPrediction(data) {
  showPrediction({...data, top_predictions: data.top_predictions || []});
  els.predictionState.title = data.method || '';
  els.predictionState.textContent = data.accepted
    ? (data.agreement === true ? 'Sign recognized — models agree' : 'Sign recognized')
    : 'Low confidence — try again';
  // Per-model breakdown chips.
  const chips = (data.sources || []).filter(source => source.ran).map(source => {
    if (source.ok) {
      return `<span class="chip ok">${MODEL_LABELS[source.model] || source.model}: <b>${source.label}</b> ${Math.round((source.confidence || 0) * 100)}%</span>`;
    }
    return `<span class="chip down" title="${source.detail}">${MODEL_LABELS[source.model] || source.model}: failed</span>`;
  });
  if (data.agreement === false) chips.push('<span class="chip warn">letter models disagreed</span>');
  els.modelChips.innerHTML = chips.join('');
  els.modelChips.hidden = !chips.length;
  // Offer the word model's candidate when the consensus landed on a letter.
  if (data.word && data.word.label !== data.label) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'word-candidate';
    button.textContent = `Word: ${data.word.label} ${Math.round(data.word.confidence * 100)}%`;
    button.onclick = () => {
      currentLabel = data.word.label;
      els.letter.textContent = currentLabel;
      els.letter.classList.add('word');
      els.addButton.disabled = false;
      els.addButton.textContent = 'Add word';
    };
    els.alternatives.hidden = false;
    if (!els.alternatives.textContent.trim()) els.alternatives.textContent = 'Maybe ';
    els.alternatives.appendChild(document.createTextNode(' '));
    els.alternatives.appendChild(button);
  }
}

async function postForm(url, form) {
  const response = await fetch(url, {method: 'POST', body: form});
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Recognition failed (HTTP ${response.status}).`);
  return data;
}

async function postJson(url, body) {
  const response = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? 'Please show one hand and try again.' : data.detail;
    throw new Error(detail || `Recognition failed (HTTP ${response.status}).`);
  }
  return data;
}

function showPrediction(data) {
  const config = MODES[mode];
  currentLabel = data.label;
  els.predictionEmpty.hidden = true;
  els.predictionResult.hidden = false;
  els.modelChips.hidden = true;
  els.letter.textContent = data.label;
  els.letter.classList.toggle('word', config.addsWord || String(data.label).length > 1);
  const percent = Math.round(data.confidence * 100);
  els.confidence.textContent = `${percent}% confidence`;
  els.confidenceMeter.style.width = `${percent}%`;
  els.predictionState.textContent = data.accepted ? 'Sign recognized' : 'Low confidence — try again';
  els.addButton.disabled = !data.accepted;
  els.addButton.textContent = config.addLabel;
  const alternatives = data.top_predictions.slice(1);
  els.alternatives.hidden = !alternatives.length;
  els.alternatives.innerHTML = alternatives.length ? `Maybe ${alternatives.map(item => `<button type="button" data-label="${item.label}">${item.label} ${Math.round(item.confidence * 100)}%</button>`).join('')}` : '';
  els.alternatives.querySelectorAll('button').forEach(button => button.onclick = () => {
    currentLabel = button.dataset.label;
    els.letter.textContent = currentLabel;
    els.addButton.disabled = false;
  });
}

function addToMessage() {
  if (!currentLabel) return;
  // In combined mode a prediction can be a letter or a whole word; words get
  // the same trailing space the word model's results use.
  const asWord = MODES[mode].addsWord || String(currentLabel).trim().length > 1;
  const addition = asWord ? `${currentLabel} ` : currentLabel;
  if (els.message.value.length + addition.length > 240) return toast('The message is full.', true);
  els.message.value += addition;
  updateMessage();
  toast(`Added ${currentLabel}`);
}

// Renders the typed message as an ordered fingerspelling sequence. Both
// bundled letter models only cover a-z, so anything else (digits, accents,
// punctuation) is shown as an explicit "cannot be fingerspelled" tile and
// summarised below, instead of being dropped without telling the user.
function updateMessage() {
  els.characterCount.textContent = `${els.message.value.length} / 240`;
  els.fingerSequence.innerHTML = '';
  els.fingerNote.textContent = '';
  if (!els.message.value) {
    els.fingerSequence.innerHTML = '<span class="sequence-empty">Your sequence will appear here</span>';
    return;
  }
  const unspellable = [];
  [...els.message.value.toLowerCase()].forEach(character => {
    const tile = document.createElement('span');
    tile.setAttribute('role', 'listitem');
    if (/[a-z]/.test(character)) {
      tile.className = 'finger-letter';
      tile.textContent = character;
      tile.title = `Sign the letter ${character.toUpperCase()}`;
      tile.setAttribute('aria-label', `Letter ${character.toUpperCase()}`);
    } else if (/\s/.test(character)) {
      tile.className = 'finger-space';
      tile.title = 'Space';
      tile.setAttribute('aria-label', 'Space');
    } else {
      tile.className = 'finger-unknown';
      tile.textContent = character;
      tile.title = `"${character}" has no letter sign in this alphabet — say or write it instead`;
      tile.setAttribute('aria-label', `${character}, no letter sign`);
      if (!unspellable.includes(character)) unspellable.push(character);
    }
    els.fingerSequence.appendChild(tile);
  });
  if (unspellable.length) {
    els.fingerNote.textContent = `No letter sign for ${unspellable.map(character => `"${character}"`).join(', ')} — these use number or non-manual signs, so say or write them instead.`;
  }
}

// --- Translation (Multilingual Output module) -----------------------------

async function loadLanguages() {
  try {
    const response = await fetch('/api/languages');
    const data = await response.json();
    (data.languages || []).forEach(language => {
      const option = document.createElement('option');
      option.value = language.code;
      option.textContent = language.name;
      els.languageSelect.appendChild(option);
    });
  } catch {
    toast('Could not load the list of languages.', true);
  }
}

async function translateMessage() {
  const text = els.message.value.trim();
  const language = els.languageSelect.value;
  if (!text) return toast('Build or type a message first.', true);
  if (!language) return toast('Choose a target language first.', true);
  els.translateButton.disabled = true;
  try {
    const response = await fetch('/api/translate', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text, language}),
    });
    const data = await response.json();
    if (data.success) {
      translatedText = data.translated;
      translatedLanguage = language;
      els.translationOutput.textContent = `${data.translated}`;
      els.speakTranslatedButton.disabled = false;
    } else {
      translatedText = '';
      els.speakTranslatedButton.disabled = true;
      els.translationOutput.textContent = data.message || 'Could not translate that text.';
    }
  } catch {
    toast('Translation request failed.', true);
  } finally {
    els.translateButton.disabled = false;
  }
}

// --- Text-to-speech (Speak / Replay / Stop / Mute) ------------------------

function speakText(text, lang) {
  if (!text || !text.trim()) return toast('Nothing to speak yet.', true);
  if (!('speechSynthesis' in window)) return toast('Speech is not supported in this browser.', true);
  if (els.muteToggle.checked) return toast('Unmute to hear speech.', true);
  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  if (lang) {
    utterance.lang = lang;
    const voice = speechSynthesis.getVoices().find(candidate => candidate.lang && candidate.lang.startsWith(lang.split('-')[0]));
    if (voice) utterance.voice = voice;
  }
  lastUtteranceText = text;
  speechSynthesis.speak(utterance);
}

// --- Emergency phrases (on-screen/spoken now; Twilio SMS + call when configured) ---

function renderEmergencyStatus(notification) {
  if (!notification) return;
  if (notification.configured) {
    const channels = [notification.sms_enabled && 'SMS', notification.call_enabled && 'voice call'].filter(Boolean);
    els.emergencyNote.innerHTML = `Alerts are delivered <strong>for real</strong> by Twilio to <strong>${notification.to_masked || 'your number'}</strong> (${channels.join(' + ') || 'no channels'}). Your current message is included in the SMS. This notifies your contact — it does <strong>not</strong> call public emergency services (112 / 911).`;
    els.emergencyStatus.hidden = false;
    els.emergencyStatus.className = 'emergency-status ok';
    els.emergencyStatus.textContent = notification.sdk_available === false
      ? 'Twilio is configured, but the server is missing the twilio package — nothing can be sent yet.'
      : `Twilio ${channels.join(' + ')} enabled → ${notification.to_masked}`;
  } else {
    els.emergencyNote.innerHTML = `Prototype only: this does <strong>not</strong> call real emergency services, police, or an ambulance, and sends no SMS. It shows and speaks a phrase loudly so a bystander can help. Configure Twilio (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, EMERGENCY_TO_NUMBER) to enable real SMS + call delivery.`;
    els.emergencyStatus.hidden = false;
    els.emergencyStatus.className = 'emergency-status warn';
    els.emergencyStatus.textContent = 'Demo mode — no SMS or call will be sent.';
  }
}

async function openEmergencyPanel() {
  els.emergencyPanel.hidden = false;
  const language = els.languageSelect.value || 'en';
  try {
    const response = await fetch(`/api/emergency/phrases?language=${encodeURIComponent(language)}`);
    const data = await response.json();
    renderEmergencyStatus(data.notification);
    els.emergencyPhrases.innerHTML = '';
    data.phrases.forEach(phrase => {
      const row = document.createElement('div');
      row.className = 'emergency-phrase';
      row.innerHTML = `<div><strong>${phrase.translated}</strong><span>${phrase.text}</span></div>`;
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = 'Show & speak';
      button.onclick = () => triggerEmergencyPhrase(phrase, language);
      row.appendChild(button);
      els.emergencyPhrases.appendChild(row);
    });
  } catch {
    els.emergencyPhrases.innerHTML = '<p>Could not load emergency phrases.</p>';
  }
}

function closeEmergencyPanel() {
  els.emergencyPanel.hidden = true;
}

function toastDelivery(notification) {
  if (!notification || !notification.configured) return;
  const sent = (notification.channels || []).filter(channel => channel.sent).map(channel => channel.channel.toUpperCase());
  if (sent.length) {
    toast(`Twilio alert sent (${sent.join(' + ')}) to ${notification.to_masked}`);
  } else if ((notification.channels || []).length) {
    toast('Twilio delivery failed — the phrase was shown and spoken only.', true);
  }
}

async function triggerEmergencyPhrase(phrase, language) {
  els.emergencyBanner.textContent = phrase.translated;
  els.emergencyBanner.hidden = false;
  if ('speechSynthesis' in window) {
    speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(phrase.translated);
    const lang = VOICE_LANGUAGE_TAGS[language];
    if (lang) utterance.lang = lang;
    speechSynthesis.speak(utterance); // Emergency speech always plays, even if muted.
  }
  setTimeout(() => { els.emergencyBanner.hidden = true; }, 6000);
  try {
    const response = await fetch('/api/emergency/alert', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        phrase_id: phrase.id,
        language,
        context_message: els.message.value.trim().slice(0, 240) || undefined,
      }),
    });
    if (response.ok) toastDelivery((await response.json()).notification);
  } catch {
    // Logging the alert is best-effort only; the on-screen/spoken phrase already happened.
  }
}

modeButtons.forEach(button => button.onclick = () => setMode(button.dataset.mode));
els.cameraButton.onclick = () => stream ? stopCamera() : startCamera();
els.pauseButton.onclick = togglePause;
els.captureButton.onclick = () => {
  if (mode === 'combined') return captureCombined();
  if (mode === 'alphabet') return captureAlphabet();
  if (mode === 'fingerspelling') return captureFingerspelling();
  if (mode === 'continuous') return toggleContinuous();
  return recordWord();
};
els.flipButton.onclick = async () => { if (recording) return; facingMode = facingMode === 'user' ? 'environment' : 'user'; if (stream) await startCamera(); };
els.addButton.onclick = addToMessage;
els.spaceButton.onclick = async () => {
  if (mode === 'continuous' && await continuousAction('/space')) return;
  if (els.message.value.length < 240) { els.message.value += ' '; updateMessage(); }
};
els.periodButton.onclick = async () => {
  if (mode === 'continuous' && await continuousAction('/punctuation')) return;
  if (els.message.value.length < 240) { els.message.value = els.message.value.replace(/\s+$/, '') + '.'; updateMessage(); }
};
els.backspaceButton.onclick = async () => {
  if (mode === 'continuous' && await continuousAction('/backspace')) return;
  els.message.value = els.message.value.slice(0, -1); updateMessage();
};
els.clearButton.onclick = async () => {
  if (mode === 'continuous' && await continuousAction('/clear')) return;
  els.message.value = ''; updateMessage();
};
els.message.oninput = updateMessage;
els.speakButton.onclick = () => speakText(els.message.value, 'en-IN');
els.replayButton.onclick = () => speakText(lastUtteranceText || els.message.value, 'en-IN');
els.stopSpeechButton.onclick = () => { if ('speechSynthesis' in window) speechSynthesis.cancel(); };
els.translateButton.onclick = translateMessage;
els.speakTranslatedButton.onclick = () => speakText(translatedText, VOICE_LANGUAGE_TAGS[translatedLanguage]);
els.emergencyButton.onclick = openEmergencyPanel;
els.emergencyClose.onclick = closeEmergencyPanel;
els.emergencyPanel.addEventListener('click', event => { if (event.target === els.emergencyPanel) closeEmergencyPanel(); });
window.addEventListener('beforeunload', stopCamera);
setMode('combined');
checkApi();
// Model availability can change after boot (lazy loads, a restarted deploy),
// so keep the status dots and mode warnings fresh instead of trusting the
// single health call made at page load.
setInterval(checkApi, 30000);
updateMessage();
loadLanguages();
