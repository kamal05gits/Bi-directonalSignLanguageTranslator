const $ = (id) => document.getElementById(id);
const els = Object.fromEntries(['apiStatus','video','canvas','cameraEmpty','cameraButton','captureButton','flipButton','predictionEmpty','predictionResult','letter','confidence','confidenceMeter','predictionState','addButton','alternatives','message','characterCount','clearButton','spaceButton','backspaceButton','speakButton','fingerSequence','toast'].map(id => [id, $(id)]));
let stream = null;
let facingMode = 'user';
let currentLetter = '';
let toastTimer;

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
  } catch {
    els.apiStatus.className = 'status offline';
    els.apiStatus.querySelector('span').textContent = 'Service unavailable';
  }
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
  els.video.srcObject = null;
  els.cameraEmpty.hidden = false;
  els.captureButton.disabled = true;
  els.cameraButton.textContent = 'Start camera';
}

async function capture() {
  if (!stream || !els.video.videoWidth) return;
  els.captureButton.disabled = true;
  els.captureButton.textContent = 'Recognizing…';
  const size = Math.min(els.video.videoWidth, els.video.videoHeight);
  const sx = (els.video.videoWidth - size) / 2;
  const sy = (els.video.videoHeight - size) / 2;
  els.canvas.width = 640;
  els.canvas.height = 640;
  const context = els.canvas.getContext('2d');
  // Mirror selfie camera to match what the user sees.
  if (facingMode === 'user') { context.translate(640, 0); context.scale(-1, 1); }
  context.drawImage(els.video, sx, sy, size, size, 0, 0, 640, 640);
  context.setTransform(1, 0, 0, 1, 0, 0);
  const blob = await new Promise(resolve => els.canvas.toBlob(resolve, 'image/jpeg', .9));
  const form = new FormData();
  form.append('file', blob, 'capture.jpg');
  try {
    const response = await fetch('/api/predict', {method: 'POST', body: form});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Recognition failed.');
    showPrediction(data);
  } catch (error) {
    toast(error.message, true);
  } finally {
    els.captureButton.disabled = false;
    els.captureButton.textContent = 'Recognize sign';
  }
}

function showPrediction(data) {
  currentLetter = data.label;
  els.predictionEmpty.hidden = true;
  els.predictionResult.hidden = false;
  els.letter.textContent = data.label;
  const percent = Math.round(data.confidence * 100);
  els.confidence.textContent = `${percent}% confidence`;
  els.confidenceMeter.style.width = `${percent}%`;
  els.predictionState.textContent = data.accepted ? 'Sign recognized' : 'Low confidence — try again';
  els.addButton.disabled = !data.accepted;
  const alternatives = data.top_predictions.slice(1);
  els.alternatives.hidden = !alternatives.length;
  els.alternatives.innerHTML = alternatives.length ? `Maybe ${alternatives.map(item => `<button type="button" data-letter="${item.label}">${item.label} ${Math.round(item.confidence * 100)}%</button>`).join('')}` : '';
  els.alternatives.querySelectorAll('button').forEach(button => button.onclick = () => {
    currentLetter = button.dataset.letter;
    els.letter.textContent = currentLetter;
    els.addButton.disabled = false;
  });
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

els.cameraButton.onclick = () => stream ? stopCamera() : startCamera();
els.captureButton.onclick = capture;
els.flipButton.onclick = async () => { facingMode = facingMode === 'user' ? 'environment' : 'user'; if (stream) await startCamera(); };
els.addButton.onclick = () => { if (currentLetter && els.message.value.length < 240) { els.message.value += currentLetter; updateMessage(); toast(`Added ${currentLetter.toUpperCase()}`); } };
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
checkApi();
updateMessage();
