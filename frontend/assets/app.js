const $ = (id) => document.getElementById(id);
const els = Object.fromEntries(['apiStatus','video','canvas','cameraEmpty','cameraButton','captureButton','flipButton','stageTitle','guideLabel','privacyNote','predictionEmpty','predictionResult','predictionHint','letter','confidence','confidenceMeter','predictionState','addButton','alternatives','message','characterCount','clearButton','spaceButton','backspaceButton','speakButton','fingerSequence','toast'].map(id => [id, $(id)]));
const modeButtons = [...document.querySelectorAll('.model-option')];
const modelDots = Object.fromEntries([...document.querySelectorAll('[data-model-dot]')].map(dot => [dot.dataset.modelDot, dot]));

const WORD_SECONDS = 3;
const HANDS_VERSION = '0.4.1675469240';
const HANDS_CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${HANDS_VERSION}`;

let stream = null;
let facingMode = 'user';
let mode = 'alphabet';
let modelHealth = {};
let currentLabel = '';
let toastTimer;
let handsPromise = null;
let handsInstance = null;
let recording = false;

const MODES = {
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
    for (const [name, dot] of Object.entries(modelDots)) {
      const health = modelHealth[name];
      const ok = Boolean(health && health.available);
      dot.className = `model-dot ${ok ? 'ok' : 'down'}`;
      dot.title = health ? health.detail : 'Unknown';
    }
    if (modelHealth.word && !modelHealth.word.available && mode === 'word') toast(modelHealth.word.detail, true);
  } catch {
    els.apiStatus.className = 'status offline';
    els.apiStatus.querySelector('span').textContent = 'Service unavailable';
  }
}

function setMode(nextMode) {
  if (recording) return toast('Stop the recording before switching models.', true);
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
    els.cameraButton.textContent = 'Stop camera';
  } catch (error) {
    toast(error.name === 'NotAllowedError' ? 'Camera permission was denied. Allow it in browser settings.' : 'Could not start your camera.', true);
  }
}

function stopCamera() {
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  recording = false;
  els.video.srcObject = null;
  els.cameraEmpty.hidden = false;
  els.captureButton.disabled = true;
  els.cameraButton.textContent = 'Start camera';
  els.countdown.hidden = true;
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

function detectHands(source, hands) {
  return new Promise((resolve, reject) => {
    hands.onResults(resolve);
    hands.send({image: source}).catch(reject);
  });
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

function pickVideoMimeType() {
  if (!window.MediaRecorder) return null;
  const candidates = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
  return candidates.find(candidate => MediaRecorder.isTypeSupported(candidate)) || '';
}

async function recordWord() {
  if (!stream || recording) return;
  const mimeType = pickVideoMimeType();
  if (mimeType === null) return toast('Video recording is not supported in this browser.', true);
  const extension = mimeType.includes('mp4') ? 'mp4' : 'webm';
  let recorder;
  try {
    recorder = new MediaRecorder(stream, mimeType ? {mimeType, videoBitsPerSecond: 2500000} : undefined);
  } catch {
    return toast('Could not start video recording in this browser.', true);
  }
  const chunks = [];
  recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
  const stopped = new Promise(resolve => recorder.onstop = resolve);
  recording = true;
  els.captureButton.disabled = true;
  els.flipButton.disabled = true;
  recorder.start(250);
  for (let remaining = WORD_SECONDS; remaining > 0 && recording; remaining--) {
    els.countdown.textContent = remaining;
    els.countdown.hidden = false;
    els.captureButton.textContent = `Recording… ${remaining}s`;
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  els.countdown.hidden = true;
  if (recorder.state !== 'inactive') recorder.stop();
  await stopped;
  recording = false;
  els.flipButton.disabled = false;
  const blob = new Blob(chunks, {type: recorder.mimeType || mimeType || 'video/webm'});
  if (!blob.size) {
    els.captureButton.disabled = false;
    els.captureButton.textContent = MODES.word.captureLabel;
    return toast('Recording came back empty. Please try again.', true);
  }
  els.captureButton.textContent = 'Recognizing…';
  const form = new FormData();
  form.append('file', blob, `sign.${extension}`);
  try {
    const data = await postForm('/api/predict/word', form);
    showPrediction(data);
  } catch (error) {
    toast(error.message, true);
  } finally {
    els.captureButton.disabled = !stream;
    els.captureButton.textContent = MODES.word.captureLabel;
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
  els.letter.textContent = data.label;
  els.letter.classList.toggle('word', config.addsWord);
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
  const addition = MODES[mode].addsWord ? `${currentLabel} ` : currentLabel;
  if (els.message.value.length + addition.length > 240) return toast('The message is full.', true);
  els.message.value += addition;
  updateMessage();
  toast(`Added ${currentLabel}`);
}

function updateMessage() {
  els.characterCount.textContent = `${els.message.value.length} / 240`;
  els.fingerSequence.innerHTML = '';
  if (!els.message.value) {
    els.fingerSequence.innerHTML = '<span class="sequence-empty">Your sequence will appear here</span>';
    return;
  }
  [...els.message.value.toLowerCase()].forEach(character => {
    const tile = document.createElement('span');
    if (/[a-z]/.test(character)) { tile.className = 'finger-letter'; tile.textContent = character; tile.title = `Sign the letter ${character.toUpperCase()}`; }
    else if (character === ' ') { tile.className = 'finger-space'; tile.title = 'Space'; }
    else return;
    els.fingerSequence.appendChild(tile);
  });
}

modeButtons.forEach(button => button.onclick = () => setMode(button.dataset.mode));
els.cameraButton.onclick = () => stream ? stopCamera() : startCamera();
els.captureButton.onclick = () => {
  if (mode === 'alphabet') return captureAlphabet();
  if (mode === 'fingerspelling') return captureFingerspelling();
  return recordWord();
};
els.flipButton.onclick = async () => { if (recording) return; facingMode = facingMode === 'user' ? 'environment' : 'user'; if (stream) await startCamera(); };
els.addButton.onclick = addToMessage;
els.spaceButton.onclick = () => { if (els.message.value.length < 240) { els.message.value += ' '; updateMessage(); } };
els.backspaceButton.onclick = () => { els.message.value = els.message.value.slice(0, -1); updateMessage(); };
els.clearButton.onclick = () => { els.message.value = ''; updateMessage(); };
els.message.oninput = updateMessage;
els.speakButton.onclick = () => {
  if (!els.message.value.trim()) return toast('Build or type a message first.', true);
  if (!('speechSynthesis' in window)) return toast('Speech is not supported in this browser.', true);
  speechSynthesis.cancel();
  speechSynthesis.speak(new SpeechSynthesisUtterance(els.message.value));
};
window.addEventListener('beforeunload', stopCamera);
setMode('alphabet');
checkApi();
updateMessage();
