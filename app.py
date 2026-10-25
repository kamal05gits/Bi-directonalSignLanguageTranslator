"""
Main Flask application for ISL Gesture Recognition System
"""
import importlib.util
import os
import sys
from pathlib import Path


def _prefer_local_venv():
    """Use the bundled virtual environment when the system interpreter is missing project deps."""
    if os.environ.get('_SIGNORA_USING_LOCAL_VENV') == '1':
        return

    project_root = Path(__file__).resolve().parent
    if sys.platform == 'win32':
        venv_python = project_root / 'venv' / 'Scripts' / 'python.exe'
    else:
        venv_python = project_root / 'venv' / 'bin' / 'python'

    if not venv_python.exists():
        return

    required_modules = ('flask', 'flask_cors', 'cv2', 'numpy', 'mediapipe', 'tensorflow', 'requests')
    missing = [module for module in required_modules if importlib.util.find_spec(module) is None]
    if not missing:
        return

    os.environ['_SIGNORA_USING_LOCAL_VENV'] = '1'
    try:
        os.execv(str(venv_python), [str(venv_python), *sys.argv])
    except OSError:
        pass


_prefer_local_venv()

# Fix Windows console encoding for emoji characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import os
import io
import re
import cv2
import json
import time
import base64
import glob
import subprocess
from datetime import datetime, timezone
from flask import Flask, render_template, jsonify, request, Response, send_file, send_from_directory, url_for
from flask_cors import CORS
import numpy as np
import mediapipe as mp
from threading import Thread
import threading
import queue
import requests
from urllib.parse import urlsplit
from src.utils.prediction_smoothing import PredictionSmoother, StablePredictionCommitter

# Import custom modules
from config import config
from src.models.gesture_recognizer import GestureRecognizer
from src.utils.tts_handler import TTSHandler
from src.utils.nlp_processor import NLPProcessor
from src.utils.performance_monitor import PerformanceMonitor

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(config['development'])
CORS(app)

# Initialize components
gesture_recognizer = None
tts_handler = TTSHandler()
nlp_processor = NLPProcessor()
performance_monitor = PerformanceMonitor()

# Initialize gesture recognizer immediately
print("🚀 Initializing ISL Gesture Recognition System...")
try:
    # Pass model path and configuration to gesture recognizer
    model_config = {
        'MAX_NUM_HANDS': config['development'].MAX_NUM_HANDS,
        'HAND_DETECTION_CONFIDENCE': config['development'].HAND_DETECTION_CONFIDENCE,
        'HAND_TRACKING_CONFIDENCE': config['development'].HAND_TRACKING_CONFIDENCE
    }
    
    gesture_recognizer = GestureRecognizer(
        model_path=config['development'].CNN_MODEL_PATH,
        config=model_config
    )
    print("✅ Gesture recognizer initialized successfully")
except Exception as e:
    print(f"❌ Error initializing gesture recognizer: {e}")
    gesture_recognizer = None

# Global variables for video streaming
camera = None
output_frame = None
lock = threading.Lock()
gesture_queue = queue.Queue()
current_sentence = ""
emergency_sos_attempts = {}
emergency_sos_lock = threading.Lock()

# Global variables for real-time gesture data
current_gesture = ""
current_confidence = 0.0
current_hand_count = 0
last_detection_time = 0
active_prediction_model = None
prediction_smoother = PredictionSmoother(
    window_size=app.config['PREDICTION_SMOOTHING_WINDOW'],
    min_stable_frames=app.config['PREDICTION_MIN_STABLE_FRAMES'],
    confidence_threshold=app.config['CONFIDENCE_THRESHOLD']
)
prediction_committer = StablePredictionCommitter(
    cooldown_seconds=app.config['PREDICTION_COMMIT_COOLDOWN'],
    same_gesture_cooldown_seconds=app.config['SAME_GESTURE_COOLDOWN']
)


def get_camera():
    """Initialize and return camera object"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
    return camera

def generate_frames(model_id='cnn'):
    """Generate video frames with gesture recognition"""
    global output_frame, current_sentence, current_gesture, current_confidence, current_hand_count
    
    print(f"📹 Starting video stream generation...")
    print(f"🤖 Gesture recognizer status: {'Loaded' if gesture_recognizer else 'Not loaded'}")
    print(f"🎯 Gesture confidence threshold: {app.config['CONFIDENCE_THRESHOLD']}")
    
    camera = get_camera()
    if not camera.isOpened():
        print("❌ Camera failed to open")
        return
    
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    while True:
        success, frame = camera.read()
        if not success:
            print("❌ Failed to read camera frame")
            break
            
        prediction_frame = frame
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Perform gesture recognition
        try:
            if gesture_recognizer:
                result = gesture_recognizer.predict_with_model(prediction_frame, model_id)
                buffering = bool(result and result.get('buffering'))
                if buffering:
                    select_prediction_model(model_id)
                    prediction_smoother.reset()
                    current_hand_count = int(result.get('hand_count', 0))
                    current_gesture = ''
                    current_confidence = 0.0
                    stable_prediction = None
                else:
                    stable_prediction = stabilize_prediction(result, model_id)

                if result and result.get('landmarks_data'):
                    frame = gesture_recognizer.draw_landmarks(
                        frame,
                        result['landmarks_data']
                    )

                hand_count = int(result.get('hand_count', 0)) if result else 0
                label = stable_prediction['gesture'] if stable_prediction else ''
                confidence = stable_prediction['confidence'] if stable_prediction else 0.0
                buffered_frames = int(result.get('buffered_frames', 0)) if result else 0
                sequence_length = int(result.get('sequence_length', 40)) if result else 40
                status_text = (
                    f'Collecting {buffered_frames}/{sequence_length} frames...'
                    if buffering
                    else label or 'No confident sign'
                )
                color = (0, 255, 0) if stable_prediction else (0, 165, 255)
                cv2.putText(frame, f"Hands: {hand_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Gesture: {status_text}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if stable_prediction and commit_stable_prediction(stable_prediction):
                    gesture_queue.put(stable_prediction['gesture'])

                cv2.putText(frame, f"Sentence: {current_sentence}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                stabilize_prediction(None, model_id)
                cv2.putText(frame, "No confident sign", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        except Exception:
            app.logger.exception('Server-camera prediction failed.')
            reset_live_prediction_state(model_id)
            if gesture_recognizer is not None:
                gesture_recognizer.reset_temporal_state()
            cv2.putText(
                frame,
                'Prediction error; retrying (see server log)',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            performance_monitor.log_fps(fps)
            start_time = end_time
            
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Store frame globally
        output_frame = frame
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def process_gesture_sequence():
    """Process sequence of gestures to form meaningful text"""
    gestures = []
    while not gesture_queue.empty():
        gestures.append(gesture_queue.get())
    
    if gestures:
        # Process gestures using NLP processor with intelligent sentence building
        # Only process the last unique gesture to avoid repetition
        if len(gestures) > 0:
            # Take only the most recent gesture
            latest_gesture = gestures[-1]
            global current_sentence
            current_sentence = nlp_processor.process_gesture_sequence([latest_gesture], current_sentence)
            return current_sentence
    
    return current_sentence

@app.route('/')
@app.route('/home')
def index():
    """Sign-to-text gesture recognition page."""
    return render_template('index.html',
                         gesture_classes=app.config['GESTURE_CLASSES'])

@app.route('/text-to-sign')
def text_to_sign():
    """Render text as a sequence of available animated ISL signs."""
    videos_dir = os.path.join(app.root_path, 'public', 'text-to-sign')
    sign_catalog = {}
    for avatar in ('Sanket', 'Girish'):
        avatar_dir = os.path.join(videos_dir, avatar)
        filenames = os.listdir(avatar_dir) if os.path.isdir(avatar_dir) else []
        sign_catalog[avatar] = [
            {
                'name': os.path.splitext(filename)[0],
                'url': url_for('text_to_sign_video', filename=f'{avatar}/{filename}')
            }
            for filename in sorted(filenames, key=str.casefold)
            if filename.casefold().endswith('.mp4')
            and os.path.isfile(os.path.join(avatar_dir, filename))
        ]
    return render_template('text_to_sign.html', sign_catalog=sign_catalog)

@app.route('/text-to-sign/videos/<path:filename>')
def text_to_sign_video(filename):
    """Serve a text-to-sign clip from the bundled video catalog."""
    return send_from_directory(
        os.path.join(app.root_path, 'public', 'text-to-sign'),
        filename,
        conditional=True
    )

def append_word_prediction(sentence, gesture):
    """Append a word-class prediction without joining it to adjacent characters."""
    word = gesture.replace('_', ' ').strip().upper()
    if not word:
        return sentence
    return f"{sentence.rstrip()} {word} ".lstrip()


def select_prediction_model(model_id):
    """Reset temporal state when switching between live recognition models."""
    global active_prediction_model
    if model_id != active_prediction_model:
        prediction_smoother.reset()
        prediction_committer.reset()
        active_prediction_model = model_id


def reset_live_prediction_state(model_id=None):
    """Clear stale output and temporal filters after missing input or inference failure."""
    global current_gesture, current_confidence, current_hand_count

    if model_id is not None:
        select_prediction_model(model_id)
    prediction_smoother.reset()
    prediction_committer.release()
    current_gesture = ''
    current_confidence = 0.0
    current_hand_count = 0


def stabilize_prediction(result, model_id):
    """Filter frame-level predictions before displaying them or committing text."""
    global current_gesture, current_confidence, current_hand_count

    select_prediction_model(model_id)
    if not result or result.get('hand_count', 0) <= 0:
        reset_live_prediction_state()
        app.logger.debug('model=%s top-three=[] displayed=', model_id)
        return None

    current_hand_count = int(result['hand_count'])
    confidence = result.get('confidence')
    if (
        model_id == 'emergency_lstm'
        and isinstance(confidence, (int, float))
        and confidence < app.config['EMERGENCY_CONFIDENCE_THRESHOLD']
    ):
        prediction_smoother.reset()
        stable = None
    else:
        stable = prediction_smoother.update(result)
    current_gesture = stable['gesture'] if stable else ''
    current_confidence = stable['confidence'] if stable else 0.0
    app.logger.debug(
        'model=%s top-three=%s displayed=%s confidence=%.4f',
        model_id,
        result.get('top_predictions', []),
        current_gesture,
        current_confidence
    )
    return stable


def commit_stable_prediction(prediction, client_ip='sign-recognition'):
    """Append one stable gesture once, requiring release before repeating it."""
    global current_sentence

    if not prediction:
        return False

    gesture = prediction['gesture']
    now = time.time()
    if not prediction_committer.should_commit(gesture, now):
        return False

    if gesture == 'SPACE':
        current_sentence += ' '
    elif prediction.get('gesture_type') == 'word':
        current_sentence = append_word_prediction(current_sentence, gesture)
    else:
        current_sentence = nlp_processor.process_gesture_sequence([gesture], current_sentence)

    if prediction.get('model_id') == 'emergency_lstm':
        prediction['emergency_detected'] = True
        if float(prediction.get('confidence', 0.0)) >= 0.65:
            sos_result, _ = dispatch_emergency_sos(
                source='Signora emergency sign recognition',
                gesture=gesture,
                client_ip=client_ip
            )
            prediction['emergency_sos'] = sos_result
            if sos_result.get('success'):
                app.logger.warning('SOS dispatched after stable emergency sign: %s.', gesture)
            else:
                app.logger.warning(
                    'Emergency sign %s was recognized, but SOS dispatch failed: %s',
                    gesture,
                    sos_result.get('error', 'unknown error')
                )
        else:
            prediction['emergency_sos'] = {
                'success': False,
                'error': 'Emergency sign confidence was below the 0.65 SOS threshold.'
            }

    app.logger.info(
        'Committed stable sign %s (confidence=%.4f).',
        gesture,
        prediction['confidence']
    )
    return True


@app.route('/api/predict_frame', methods=['POST'])
def predict_frame():
    """Accept a base64 webcam frame from the browser and return gesture prediction.
    This enables client-side camera (needed for cloud deployment where no server camera exists).
    """
    global current_sentence, current_gesture, current_confidence, current_hand_count
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Request body must be a JSON object.'}), 400

        image_data = data.get('image', '')
        if not isinstance(image_data, str) or not image_data:
            return jsonify({'error': 'No image data'}), 400

        # Decode base64 → OpenCV image
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        try:
            img_bytes = base64.b64decode(image_data, validate=True)
        except (ValueError, base64.binascii.Error):
            return jsonify({'error': 'Image data is not valid base64.'}), 400
        if not img_bytes:
            return jsonify({'error': 'No image data'}), 400
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        model_id = data.get('model_id', 'cnn')
        if gesture_recognizer is None:
            return jsonify({'error': 'No gesture recognition model is loaded.'}), 503
        if not isinstance(model_id, str):
            return jsonify({'error': 'Model id must be a string.'}), 400
        selectable_models = {
            model['id'] for model in gesture_recognizer.get_available_models()
            if model['selectable']
        }
        if model_id not in selectable_models:
            return jsonify({'error': f"Model '{model_id}' is not available for live sign-to-text."}), 400

        # Run gesture recognition
        result = gesture_recognizer.predict_with_model(frame, model_id)
        result_data = {
            'gesture': '',
            'confidence': 0.0,
            'candidate_gesture': (
                result.get('gesture', '')
                if result and isinstance(result.get('gesture'), str)
                else ''
            ),
            'candidate_confidence': (
                float(result.get('confidence', 0.0))
                if result and isinstance(result.get('confidence'), (int, float))
                else 0.0
            ),
            'hand_count': 0,
            'stable': False,
            'top_predictions': [],
            'model_id': model_id,
            'model_buffering': bool(result and result.get('buffering')),
            'buffered_frames': int(result.get('buffered_frames', 0)) if result else 0,
            'sequence_length': int(result.get('sequence_length', 0)) if result else 0
        }
        if result_data['model_buffering']:
            select_prediction_model(model_id)
            current_gesture = ''
            current_confidence = 0.0
            current_hand_count = int(result.get('hand_count', 0))
            prediction_smoother.reset()
            app.logger.debug(
                'model=%s top-three=[] displayed= (buffering)',
                model_id
            )
            result_data['hand_count'] = current_hand_count
        else:
            stable_prediction = stabilize_prediction(result, model_id)
            if result:
                result_data['hand_count'] = int(result.get('hand_count', 0))
                result_data['top_predictions'] = result.get('top_predictions', [])

            if stable_prediction:
                result_data.update({
                    'gesture': stable_prediction['gesture'],
                    'confidence': stable_prediction['confidence'],
                    'gesture_type': stable_prediction.get('gesture_type', 'character'),
                    'stable': True
                })
                if data.get('build_sentence') and commit_stable_prediction(
                    stable_prediction,
                    request.remote_addr or 'unknown'
                ):
                    result_data['sentence'] = current_sentence
                    if stable_prediction.get('emergency_detected'):
                        result_data['emergency_detected'] = True
                        result_data['emergency_sos'] = stable_prediction['emergency_sos']

        return jsonify(result_data)
    except Exception:
        app.logger.exception('Frame prediction failed.')
        reset_live_prediction_state(locals().get('model_id'))
        if gesture_recognizer is not None:
            gesture_recognizer.reset_temporal_state()
        return jsonify({
            'error': 'Prediction failed. The prediction state was reset; check the server log for details.'
        }), 500

@app.route('/api/translate', methods=['POST'])
def translate_text():
    """Translate text into a supported voice-assistance language."""
    try:
        data = request.get_json(silent=True) or {}
        raw_text = data.get('text', '')
        text = raw_text.strip() if isinstance(raw_text, str) else ''
        target_lang = data.get('lang', 'ta')

        if not text:
            return jsonify({'success': False, 'error': 'No text provided'}), 400

        supported_languages = {'en', 'ta', 'te', 'hi', 'kn', 'ml', 'bn', 'ur'}
        if not isinstance(target_lang, str) or target_lang not in supported_languages:
            return jsonify({
                'success': False,
                'error': 'Unsupported language. Choose English, Tamil, Telugu, Hindi, Kannada, Malayalam, Bengali, or Urdu.'
            }), 400

        try:
            from googletrans import Translator
            translator = Translator()
            result = translator.translate(text, dest=target_lang)
            return jsonify({
                'success': True,
                'original': text,
                'translated': result.text,
                'lang': target_lang
            })
        except ImportError:
            return jsonify({'success': False, 'error': 'Translation is unavailable because googletrans is not installed.'}), 503
        except Exception as e:
            return jsonify({'success': False, 'error': f'Translation failed: {str(e)}'}), 502
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/speech-audio', methods=['POST'])
def generate_speech_audio():
    """Generate browser-playable speech when a local voice is unavailable."""
    data = request.get_json(silent=True) or {}  
    raw_text = data.get('text', '')
    text = raw_text.strip() if isinstance(raw_text, str) else ''
    language = data.get('lang', '')
    supported_languages = {'en', 'ta', 'te', 'hi', 'kn', 'ml', 'bn', 'ur'}

    if not text:
        return jsonify({'success': False, 'error': 'No text provided for speech.'}), 400
    if not isinstance(language, str) or language not in supported_languages:
        return jsonify({'success': False, 'error': 'Unsupported speech language.'}), 400

    try:
        from gtts import gTTS

        audio = io.BytesIO()
        gTTS(text=text, lang=language, slow=False).write_to_fp(audio)
        audio.seek(0)
        return send_file(audio, mimetype='audio/mpeg', download_name='speech.mp3')
    except Exception as error:
        app.logger.warning('Speech audio generation failed (%s).', type(error).__name__)
        return jsonify({
            'success': False,
            'error': 'Speech audio could not be generated. Check the network connection and try again.'
        }), 502

@app.route('/api/emergency-alert', methods=['POST'])
@app.route('/api/emergency-sos', methods=['POST'])
def send_emergency_sos():
    """Forward a user-requested emergency SOS to the configured webhook."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'success': False, 'error': 'Invalid request.'}), 400

    origin = urlsplit(request.headers.get('Origin', ''))
    origin_host = (origin.hostname or '').casefold()
    request_host = request.host.split(':', 1)[0].casefold()
    if (
        origin.scheme not in {'http', 'https'}
        or not origin_host
        or origin_host != request_host
    ):
        return jsonify({'success': False, 'error': 'Request origin is not allowed.'}), 403

    result, status_code = dispatch_emergency_sos(
        source='Signora',
        client_ip=request.remote_addr or 'unknown'
    )
    return jsonify(result), status_code


def dispatch_emergency_sos(source, gesture=None, client_ip=None):
    """Send an authenticated SOS using the configured webhook and shared cooldown."""
    global emergency_sos_attempts

    webhook_url = os.environ.get('WEBHOOK_URL', '').strip()
    try:
        webhook = urlsplit(webhook_url)
        valid_webhook = (
            webhook.scheme == 'https'
            and bool(webhook.hostname)
            and not webhook.username
            and not webhook.password
            and not webhook.fragment
        )
    except ValueError:
        valid_webhook = False
    if not valid_webhook:
        return {'success': False, 'error': 'A valid HTTPS emergency webhook is not configured.'}, 503

    webhook_token = os.environ.get('WEBHOOK_TOKEN', '').strip()
    if not webhook_token:
        return {'success': False, 'error': 'Emergency webhook authentication is not configured.'}, 503

    client_ip = client_ip or 'unknown'
    now = time.time()
    with emergency_sos_lock:
        emergency_sos_attempts = {
            ip: attempted_at
            for ip, attempted_at in emergency_sos_attempts.items()
            if now - attempted_at < 3600
        }
        previous_attempt = emergency_sos_attempts.get(client_ip)
        if previous_attempt is not None and now - previous_attempt < 60:
            return ({
                'success': False,
                'error': 'Please wait before sending another emergency sos.'
            }, 429)
        emergency_sos_attempts[client_ip] = now

    try:
        payload = {
            'event': 'emergency_sos',
            'confirmed': source == 'Signora',
            'source': source,
            'message': 'I need help. Please contact me as soon as possible.',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        if gesture:
            payload['gesture'] = gesture
        response = requests.post(
            webhook_url,
            headers={
                'Authorization': f'Bearer {webhook_token}',
                'X-Webhook-Token': webhook_token
            },
            json=payload,
            timeout=(5, 15)
        )
        response.raise_for_status()
        
    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response is not None else 502
        app.logger.warning('Emergency webhook rejected alert (HTTP %s).', status_code)
        if status_code in {401, 403}:
            message = 'Webhook rejected authentication (HTTP {0}). Check WEBHOOK_TOKEN and the webhook authentication settings.'.format(status_code)
        elif status_code == 404:
            message = 'Webhook endpoint was not found (HTTP 404). Check WEBHOOK_URL.'
        elif status_code == 405:
            message = 'Webhook does not accept POST requests (HTTP 405). Check the webhook method settings.'
        elif status_code == 429:
            message = 'Webhook rate limit reached (HTTP 429). Try again later.'
        else:
            message = 'Webhook returned HTTP {0}. Check the webhook service configuration.'.format(status_code)
        return {'success': False, 'error': message}, 502
    except requests.RequestException as error:
        app.logger.warning('Emergency webhook request failed (%s).', type(error).__name__)
        return ({
            'success': False,
            'error': 'Could not connect to the emergency webhook. Check the webhook service/network and try again later.'
        }, 502)

    return {'success': True, 'message': 'Emergency alert sent.'}, 200

@app.route('/api/abbreviate', methods=['POST'])
def abbreviate_text():
    """Convert sentence to SMS-style abbreviation for testing"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        # Simple abbreviation rules
        abbrevs = {
            'are': 'r', 'you': 'u', 'to': '2', 'for': '4', 'be': 'b',
            'see': 'c', 'why': 'y', 'too': '2', 'before': 'b4',
            'great': 'gr8', 'late': 'l8', 'wait': 'w8', 'please': 'pls',
            'because': 'bcz', 'people': 'ppl', 'about': 'abt',
            'with': 'wth', 'have': 'hv', 'your': 'ur', 'tomorrow': '2mro',
            'tonight': '2nite', 'today': '2day', 'the': 'd', 'we': 'v',
            'and': 'n', 'okay': 'ok', 'thanks': 'thx', 'never': 'nvr',
        }
        
        words = text.lower().split()
        result = []
        for w in words:
            result.append(abbrevs.get(w, w))
        
        return jsonify({'success': True, 'abbreviated': ' '.join(result), 'original': text})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/data_image/<path:filepath>')
def serve_data_image(filepath):
    """Serve images from the data/raw/gestures directory"""
    from flask import send_from_directory
    return send_from_directory('data/raw/gestures', filepath)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    model_id = request.args.get('model_id', 'cnn')
    if gesture_recognizer is None or model_id not in {
        model['id'] for model in gesture_recognizer.get_available_models()
        if model['selectable']
    }:
        return jsonify({'error': f"Model '{model_id}' is not available for live sign-to-text."}), 400
    return Response(generate_frames(model_id),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/gesture_data')
def get_gesture_data():
    """API endpoint to get current gesture recognition data"""
    try:
        data = {
            'current_sentence': current_sentence,
            'current_gesture': current_gesture,
            'confidence': current_confidence,
            'hand_count': current_hand_count,
            'fps': performance_monitor.get_current_fps(),
            'last_detection_time': last_detection_time,
            'timestamp': time.time()
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/api/speak', methods=['POST'])
def speak_text():
    """API endpoint for text-to-speech conversion"""
    try:
        data = request.get_json()
        text = data.get('text', current_sentence)

        if text and text.strip():
            print(f"🗣️ Speaking text: '{text}'")
            print(f"🔊 TTS Handler available: {tts_handler.is_available()}")
            print(f"🔊 TTS Engine type: {tts_handler.engine_type}")
            success = tts_handler.speak(text.strip())
            print(f"✅ TTS result: {success}")
            return jsonify({'success': success, 'text': text})
        else:
            print("⚠️ No text provided for TTS")
            return jsonify({'error': 'No text to speak'}), 400

    except Exception as e:
        print(f"❌ TTS Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/suggestions', methods=['POST'])
def get_suggestions():
    """AI-powered word suggestions endpoint.
    Expands abbreviations, completes partial words, and suggests next words."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'suggestions': []})
        
        suggestions = []
        words = text.split()
        last_word = words[-1].lower() if words else ''
        
        # --- 1. Abbreviation / Shorthand expansion ---
        abbreviations = {
            'u': 'you', 'r': 'are', 'v': 'very', 'y': 'why',
            'n': 'and', 'b': 'be', 'c': 'see', 'k': 'okay',
            'w': 'with', 'd': 'the', 'm': 'am', 'i': 'I',
            'ur': 'your', 'bc': 'because', 'bt': 'but',
            'hw': 'how', 'wt': 'what', 'wr': 'where',
            'gd': 'good', 'gm': 'good morning', 'gn': 'good night',
            'tq': 'thank you', 'ty': 'thank you', 'thx': 'thanks',
            'pls': 'please', 'plz': 'please', 'msg': 'message',
            'tmr': 'tomorrow', 'tdy': 'today', 'yday': 'yesterday',
            'brb': 'be right back', 'idk': "I don't know",
            'imo': 'in my opinion', 'lol': 'laughing out loud',
        }
        
        if last_word in abbreviations:
            suggestions.append(abbreviations[last_word])
        
        # --- 2. Partial word completion ---
        common_words = [
            'hello', 'help', 'happy', 'have', 'how', 'here', 'home',
            'thank', 'thanks', 'the', 'this', 'that', 'they', 'there', 'their', 'time', 'today', 'tomorrow',
            'good', 'great', 'go', 'going', 'give', 'get',
            'please', 'play', 'place',
            'want', 'water', 'what', 'when', 'where', 'which', 'why', 'with', 'will', 'work',
            'need', 'name', 'nice', 'new', 'now', 'never', 'next', 'night',
            'morning', 'more', 'make', 'much', 'many', 'meet', 'my',
            'like', 'love', 'live', 'learn', 'let', 'look', 'long',
            'come', 'call', 'can', 'care',
            'food', 'feel', 'find', 'friend', 'from', 'fine', 'family',
            'very', 'visit',
            'school', 'sorry', 'some', 'stop', 'start', 'study',
            'eat', 'evening', 'every', 'enough',
            'know', 'keep',
            'yes', 'you', 'your',
            'are', 'about', 'after', 'again', 'also',
            'because', 'before', 'best', 'better', 'between',
            'day', 'do', 'done',
            'okay', 'open', 'only', 'over',
        ]
        
        if len(last_word) >= 2 and last_word not in abbreviations:
            completions = [w for w in common_words 
                          if w.startswith(last_word) and w != last_word]
            suggestions.extend(completions[:4])
        
        # --- 3. Next-word prediction based on context ---
        next_word_map = {
            'i': ['am', 'want', 'need', 'like', 'love', 'have', 'will', 'can'],
            'you': ['are', 'can', 'will', 'have', 'need', 'want', 'should'],
            'how': ['are', 'do', 'can', 'much', 'many', 'about'],
            'what': ['is', 'are', 'do', 'time', 'about', 'happened'],
            'thank': ['you', 'god'],
            'good': ['morning', 'afternoon', 'evening', 'night', 'job', 'day'],
            'nice': ['to', 'meeting', 'day', 'work'],
            'please': ['help', 'come', 'give', 'call', 'wait', 'stop'],
            'can': ['you', 'I', 'we', 'help'],
            'i am': ['fine', 'good', 'happy', 'sorry', 'here', 'going'],
            'i want': ['to', 'food', 'water', 'help'],
            'i need': ['help', 'water', 'food', 'time', 'you', 'to'],
            'i like': ['you', 'this', 'it', 'food'],
            'i love': ['you', 'this', 'it'],
            'i will': ['come', 'go', 'help', 'call', 'do'],
            'do you': ['want', 'need', 'know', 'like', 'have'],
            'are you': ['okay', 'fine', 'happy', 'coming', 'ready', 'here'],
            'my name': ['is'],
            'very': ['much', 'good', 'happy', 'nice', 'well'],
            'see': ['you', 'this', 'that'],
            'where': ['are', 'is', 'do'],
            'we': ['are', 'can', 'will', 'should', 'need'],
            'he': ['is', 'has', 'can', 'will'],
            'she': ['is', 'has', 'can', 'will'],
            'it': ['is', 'was', 'will'],
            'they': ['are', 'have', 'can', 'will'],
            'let': ['me', 'us', 'go'],
            'come': ['here', 'home', 'back'],
        }
        
        # Check multi-word context first (last 2 words), then single word
        if len(words) >= 2:
            two_word_ctx = ' '.join(words[-2:]).lower()
            if two_word_ctx in next_word_map:
                suggestions.extend(next_word_map[two_word_ctx])
        
        if last_word in next_word_map and last_word not in abbreviations:
            for w in next_word_map[last_word]:
                if w not in suggestions:
                    suggestions.append(w)
        
        # Deduplicate and limit
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s.lower() not in seen:
                seen.add(s.lower())
                unique_suggestions.append(s)
        
        return jsonify({'suggestions': unique_suggestions[:8]})
    
    except Exception as e:
        print(f"❌ Suggestions Error: {str(e)}")
        return jsonify({'error': str(e), 'suggestions': []}), 500



@app.route('/api/clear_sentence', methods=['POST'])
def clear_sentence():
    """API endpoint to clear current sentence"""
    global current_sentence, current_gesture, current_confidence, current_hand_count
    global active_prediction_model
    current_sentence = ""
    current_gesture = ""
    current_confidence = 0.0
    current_hand_count = 0
    active_prediction_model = None
    prediction_smoother.reset()
    prediction_committer.reset()
    
    # Clear gesture queue
    while not gesture_queue.empty():
        gesture_queue.get()
    
    # Clear NLP processor buffers
    nlp_processor.clear_buffers()
    
    return jsonify({'success': True, 'message': 'Sentence cleared'})

@app.route('/api/update_sentence', methods=['POST'])
def update_sentence():
    """API endpoint to update sentence (used by backspace)"""
    global current_sentence
    try:
        data = request.get_json()
        new_sentence = data.get('sentence', '')
        current_sentence = new_sentence
        return jsonify({'success': True, 'sentence': current_sentence})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/performance')
def get_performance_data():
    """API endpoint to get performance metrics"""
    try:
        metrics = performance_monitor.get_metrics()
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/gesture_classes')
def get_gesture_classes():
    """API endpoint to get available gesture classes"""
    return jsonify({
        'classes': app.config['GESTURE_CLASSES'],
        'num_classes': app.config['NUM_CLASSES']
    })

@app.route('/api/model_info')
def get_model_info():
    """API endpoint to get model information"""
    try:
        if gesture_recognizer:
            model_info = gesture_recognizer.get_model_info()
            return jsonify(model_info)
        else:
            return jsonify({'error': 'Model not loaded'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models')
def get_models():
    """List recognition models currently ready for live translation."""
    if gesture_recognizer is None:
        return jsonify({
            'models': [],
            'model_status': [],
            'default_model': None,
            'error': 'Gesture recognizer is not initialized.'
        }), 503
    models = gesture_recognizer.get_available_models()
    return jsonify({
        'models': models,
        'model_status': gesture_recognizer.get_model_status(),
        'default_model': next(
            (model['id'] for model in models if model['id'] == 'cnn' and model['selectable']),
            next((model['id'] for model in models if model['selectable']), None)
        )
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'gesture_recognizer': gesture_recognizer is not None,
        'camera_available': camera is not None,
        'tts_available': tts_handler.is_available(),
        'timestamp': time.time()
    }
    return jsonify(status)

@app.route('/debug')
def debug_info():
    """Debug information endpoint"""
    debug_data = {
        'gesture_recognizer_loaded': gesture_recognizer is not None,
        'camera_initialized': camera is not None,
        'model_path': app.config['CNN_MODEL_PATH'],
        'confidence_threshold': app.config['CONFIDENCE_THRESHOLD'],
        'prediction_smoothing_window': app.config['PREDICTION_SMOOTHING_WINDOW'],
        'prediction_min_stable_frames': app.config['PREDICTION_MIN_STABLE_FRAMES'],
        'prediction_commit_cooldown': app.config['PREDICTION_COMMIT_COOLDOWN'],
        'same_gesture_cooldown': app.config['SAME_GESTURE_COOLDOWN'],
        'hand_detection_confidence': app.config['HAND_DETECTION_CONFIDENCE'],
        'hand_tracking_confidence': app.config['HAND_TRACKING_CONFIDENCE']
    }
    
    if gesture_recognizer:
        try:
            model_info = gesture_recognizer.get_model_info()
            debug_data['model_info'] = model_info
        except Exception as e:
            debug_data['model_error'] = str(e)
    
    return jsonify(debug_data)

# ══════════════════════════════════════════════
# DATA COLLECTION & RETRAINING APIs
# ══════════════════════════════════════════════

GESTURE_DATA_DIR = os.path.join('data', 'raw', 'gestures')

@app.route('/api/collect_gesture', methods=['POST'])
def collect_gesture():
    """Save a captured webcam frame for a gesture class"""
    try:
        data = request.get_json()
        gesture = data.get('gesture', '').upper()
        image_data = data.get('image', '')

        if not gesture or not image_data:
            return jsonify({'error': 'Missing gesture or image data'}), 400

        # Create directory for this gesture
        gesture_dir = os.path.join(GESTURE_DATA_DIR, gesture)
        os.makedirs(gesture_dir, exist_ok=True)

        # Decode base64 image
        header, encoded = image_data.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Save with timestamp filename
        timestamp = int(time.time() * 1000)
        filename = f"{gesture}_{timestamp}.jpg"
        filepath = os.path.join(gesture_dir, filename)
        cv2.imwrite(filepath, img)

        # Count total images for this gesture
        count = len([f for f in os.listdir(gesture_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        return jsonify({'success': True, 'count': count, 'file': filename})

    except Exception as e:
        print(f"❌ Collect gesture error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/capture_frame', methods=['POST'])
def capture_frame():
    """Capture current camera frame directly on server — no canvas needed"""
    try:
        data = request.get_json()
        gesture = data.get('gesture', '').upper()

        if not gesture:
            return jsonify({'error': 'Missing gesture name'}), 400

        # Use the shared camera object
        cam = get_camera()
        if cam is None or not cam.isOpened():
            return jsonify({'error': 'Camera not available'}), 500

        ret, frame = cam.read()
        if not ret or frame is None:
            return jsonify({'error': 'Failed to capture frame'}), 500

        # Flip for mirror effect (same as video_feed)
        frame = cv2.flip(frame, 1)

        # Create directory & save
        gesture_dir = os.path.join(GESTURE_DATA_DIR, gesture)
        os.makedirs(gesture_dir, exist_ok=True)

        timestamp = int(time.time() * 1000)
        filename = f"{gesture}_{timestamp}.jpg"
        filepath = os.path.join(gesture_dir, filename)
        cv2.imwrite(filepath, frame)

        count = len([f for f in os.listdir(gesture_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        return jsonify({'success': True, 'count': count, 'file': filename})

    except Exception as e:
        print(f"❌ Capture frame error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/gesture_count/<gesture>')
def gesture_count(gesture):
    """Get image count for a specific gesture"""
    gesture_dir = os.path.join(GESTURE_DATA_DIR, gesture.upper())
    if os.path.exists(gesture_dir):
        count = len([f for f in os.listdir(gesture_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    else:
        count = 0
    return jsonify({'gesture': gesture, 'count': count})

@app.route('/api/gesture_counts')
def gesture_counts():
    """Get image counts for all gestures"""
    counts = {}
    if os.path.exists(GESTURE_DATA_DIR):
        for folder in os.listdir(GESTURE_DATA_DIR):
            folder_path = os.path.join(GESTURE_DATA_DIR, folder)
            if os.path.isdir(folder_path):
                c = len([f for f in os.listdir(folder_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                counts[folder] = c
    return jsonify({'counts': counts})

@app.route('/api/gesture_samples/<gesture>')
def gesture_samples(gesture):
    """Get sample image paths for a gesture (last 20)"""
    gesture_dir = os.path.join(GESTURE_DATA_DIR, gesture.upper())
    samples = []
    if os.path.exists(gesture_dir):
        files = sorted(
            [f for f in os.listdir(gesture_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
            reverse=True
        )[:20]
        for f in files:
            samples.append(f"/data_image/{gesture.upper()}/{f}")
    return jsonify({'samples': samples})

@app.route('/api/preprocess', methods=['POST'])
def preprocess_data():
    """Run data preprocessing pipeline"""
    try:
        result = subprocess.run(
            ['python', '-m', 'src.preprocessing.preprocess_data'],
            capture_output=True, text=True, timeout=900, encoding='utf-8', errors='replace'
        )
        if result.returncode == 0:
            return jsonify({'success': True, 'message': 'Preprocessing complete!', 'output': result.stdout})
        else:
            return jsonify({'success': False, 'error': result.stderr or 'Preprocessing failed'})
    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': 'Preprocessing timed out (15 min limit)'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    """Retrain the gesture recognition model"""
    try:
        result = subprocess.run(
            ['python', '-m', 'src.models.train_cnn'],
            capture_output=True, text=True, timeout=1200, encoding='utf-8', errors='replace'
        )
        if result.returncode == 0:
            return jsonify({'success': True, 'message': 'Model retrained successfully! Restart the server to use the new model.', 'output': result.stdout})
        else:
            return jsonify({'success': False, 'error': result.stderr or 'Training failed'})
    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': 'Training timed out (20 min limit)'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ══════════════════════════════════════════════
# LSTM DYNAMIC GESTURE APIs
# ══════════════════════════════════════════════

SEQUENCE_DATA_DIR = os.path.join('data', 'sequences')
SEQUENCE_LENGTH = 30

@app.route('/api/collect_sequence', methods=['POST'])
def collect_sequence():
    """Save a 30-frame landmark sequence for LSTM training"""
    try:
        data = request.get_json()
        gesture = data.get('gesture', '').upper().strip()
        use_buffer = data.get('use_buffer', False)
        
        if not gesture:
            return jsonify({'success': False, 'error': 'Need gesture name'})
        
        if use_buffer:
            # Read from gesture recognizer's landmark buffer
            if gesture_recognizer is None or len(gesture_recognizer.landmark_buffer) < SEQUENCE_LENGTH:
                return jsonify({'success': False, 'error': f'Not enough frames buffered ({len(gesture_recognizer.landmark_buffer) if gesture_recognizer else 0}/{SEQUENCE_LENGTH}). Make sure webcam is running.'})
            seq = np.array(list(gesture_recognizer.landmark_buffer), dtype=np.float32)
        else:
            frames = data.get('frames', [])
            if len(frames) != SEQUENCE_LENGTH:
                return jsonify({'success': False, 'error': f'Need {SEQUENCE_LENGTH} frames, got {len(frames)}'})
            seq = np.array(frames, dtype=np.float32)

        if seq.shape != (SEQUENCE_LENGTH, 63) or not np.all(np.isfinite(seq)):
            return jsonify({
                'success': False,
                'error': 'Each LSTM sequence must contain 30 frames of 63 finite features.'
            }), 400

        # Save as .npy
        gesture_dir = os.path.join(SEQUENCE_DATA_DIR, gesture)
        os.makedirs(gesture_dir, exist_ok=True)
        
        existing = len([f for f in os.listdir(gesture_dir) if f.endswith('.npy')])
        np.save(os.path.join(gesture_dir, f'{gesture}_{existing}.npy'), seq)
        
        return jsonify({'success': True, 'count': existing + 1, 'gesture': gesture})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/sequence_counts')
def sequence_counts():
    """Get sequence counts per gesture class"""
    counts = {}
    if os.path.exists(SEQUENCE_DATA_DIR):
        for gesture in os.listdir(SEQUENCE_DATA_DIR):
            gesture_dir = os.path.join(SEQUENCE_DATA_DIR, gesture)
            if os.path.isdir(gesture_dir):
                counts[gesture] = len([f for f in os.listdir(gesture_dir) if f.endswith('.npy')])
    return jsonify({'counts': counts})

@app.route('/api/retrain_lstm', methods=['POST'])
def retrain_lstm():
    """Train LSTM model on collected sequences"""
    try:
        result = subprocess.run(
            ['python', '-m', 'src.models.train_lstm'],
            capture_output=True, text=True, timeout=600, encoding='utf-8', errors='replace'
        )
        if result.returncode == 0:
            return jsonify({'success': True, 'message': 'LSTM trained! Restart server to use.', 'output': result.stdout})
        else:
            return jsonify({'success': False, 'error': result.stderr or 'LSTM training failed'})
    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': 'LSTM training timed out'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('errors/500.html'), 500

if __name__ == '__main__':
    print("🚀 Initializing ISL Gesture Recognition System...")
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    print(f"🎯 Confidence threshold: {app.config['CONFIDENCE_THRESHOLD']}")
    print(f"📊 Number of gesture classes: {app.config['NUM_CLASSES']}")
    
    # Start the Flask application
    print("🌐 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG'],
        threaded=True
    )
