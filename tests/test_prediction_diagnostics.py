import json
import pickle
import base64
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from tensorflow import keras

from src.models.gesture_recognizer import GestureRecognizer
from src.preprocessing.preprocess_data import GestureDataPreprocessor
from src.utils.prediction_smoothing import PredictionSmoother, StablePredictionCommitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_saved_cnn_output_shape_matches_saved_class_mapping():
    model_path = PROJECT_ROOT / 'models' / 'trained' / 'cnn_model.h5'
    labels_path = PROJECT_ROOT / 'models' / 'trained' / 'cnn_model_labels.pkl'
    metadata_path = PROJECT_ROOT / 'models' / 'trained' / 'cnn_model_config.json'
    scaler_path = PROJECT_ROOT / 'models' / 'trained' / 'cnn_model_scaler.pkl'

    model = keras.models.load_model(model_path)
    with labels_path.open('rb') as label_file:
        encoder = pickle.load(label_file)
    with scaler_path.open('rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    labels = encoder.classes_.tolist() if hasattr(encoder, 'classes_') else list(encoder)

    assert model.input_shape == (None, 63)
    assert model.output_shape == (None, 61)
    assert scaler.n_features_in_ == 63
    assert len(labels) == model.output_shape[-1]
    assert labels == metadata['class_names']
    assert model.predict(np.zeros((1, 63), dtype=np.float32), verbose=0).shape == (1, len(labels))


def test_bundled_bigru_shape_and_indexed_class_mapping():
    import app as app_module

    recognizer = app_module.gesture_recognizer
    assert recognizer.bigru_model is not None
    assert recognizer.bigru_model.input_shape == (None, 40, 225)
    assert recognizer.bigru_model.output_shape == (None, 262)

    class_map = json.loads(
        (PROJECT_ROOT / 'models' / 'Swatarnik Dataset' / 'classes.json')
        .read_text(encoding='utf-8')
    )
    expected_labels = [class_map[str(index)].upper() for index in range(262)]
    assert recognizer.bigru_labels == expected_labels


def test_training_and_inference_landmark_normalization_match():
    raw_landmarks = np.linspace(0.01, 0.99, 63, dtype=np.float32)
    preprocessor = GestureDataPreprocessor.__new__(GestureDataPreprocessor)
    recognizer = GestureRecognizer.__new__(GestureRecognizer)
    with (PROJECT_ROOT / 'models' / 'trained' / 'cnn_model_scaler.pkl').open('rb') as scaler_file:
        recognizer.scaler = pickle.load(scaler_file)

    training_features = np.asarray(
        preprocessor.normalize_landmarks_func(raw_landmarks.tolist()),
        dtype=np.float32
    )
    inference_features = recognizer.normalize_landmarks(raw_landmarks.copy())

    np.testing.assert_allclose(training_features, inference_features, rtol=1e-6, atol=1e-6)
    live_features = recognizer.preprocess_landmarks({'landmarks': [raw_landmarks.tolist()]})
    expected_features = recognizer.scaler.transform(training_features.reshape(1, -1)).flatten()
    np.testing.assert_allclose(live_features, expected_features, rtol=1e-6, atol=1e-6)


def test_smoother_requires_confident_consensus_and_resets_on_no_hand():
    smoother = PredictionSmoother(
        window_size=5,
        min_stable_frames=3,
        confidence_threshold=0.6
    )
    assert smoother.update({'gesture': 'A', 'confidence': 0.9, 'hand_count': 1}) is None
    assert smoother.update({'gesture': 'B', 'confidence': 0.9, 'hand_count': 1}) is None
    assert smoother.update({'gesture': 'A', 'confidence': 0.8, 'hand_count': 1}) is None

    stable = smoother.update({'gesture': 'A', 'confidence': 0.9, 'hand_count': 1})
    assert stable is not None
    assert stable['gesture'] == 'A'
    assert stable['stable_frames'] == 3

    assert smoother.update({'gesture': 'A', 'confidence': 0.6, 'hand_count': 1}) is None
    assert smoother.update({'gesture': 'A', 'confidence': 0.9, 'hand_count': 1}) is None
    assert smoother.update({'gesture': 'A', 'confidence': 0.9, 'hand_count': 1}) is None
    assert smoother.update({'gesture': 'A', 'confidence': 0.9, 'hand_count': 1}) is not None
    assert smoother.update({'gesture': 'A', 'confidence': 0.9, 'hand_count': 0}) is None
    assert smoother.update({'gesture': 'A', 'confidence': 0.9, 'hand_count': 1}) is None


def test_emergency_predictions_require_higher_confidence_and_consensus():
    import app as app_module

    app_module.prediction_smoother.reset()
    app_module.active_prediction_model = None
    weak_prediction = {
        'gesture': 'HELP',
        'confidence': 0.64,
        'hand_count': 1,
        'model_id': 'emergency_lstm'
    }
    confident_prediction = {**weak_prediction, 'confidence': 0.65}

    try:
        assert all(
            app_module.stabilize_prediction(weak_prediction, 'emergency_lstm') is None
            for _ in range(3)
        )
        assert app_module.stabilize_prediction(
            confident_prediction,
            'emergency_lstm'
        ) is None
        assert app_module.stabilize_prediction(
            confident_prediction,
            'emergency_lstm'
        ) is None
        stable = app_module.stabilize_prediction(
            confident_prediction,
            'emergency_lstm'
        )
        assert stable is not None
        assert stable['gesture'] == 'HELP'
        assert stable['confidence'] == pytest.approx(0.65)
    finally:
        app_module.current_gesture = ''
        app_module.current_confidence = 0.0
        app_module.current_hand_count = 0
        app_module.active_prediction_model = None
        app_module.prediction_smoother.reset()


def test_bigru_preserves_rolling_window_during_brief_hand_tracking_loss():
    class EmptyHolistic:
        def process(self, image):
            return SimpleNamespace(
                pose_landmarks=None,
                left_hand_landmarks=None,
                right_hand_landmarks=None
            )

    recognizer = GestureRecognizer.__new__(GestureRecognizer)
    recognizer.bigru_model = object()
    recognizer.bigru_holistic = EmptyHolistic()
    recognizer.bigru_sequence_length = 40
    recognizer.bigru_buffer = deque(
        [np.ones(225, dtype=np.float32) for _ in range(40)],
        maxlen=40
    )
    recognizer.bigru_missing_hand_frames = 0

    result = recognizer._predict_bigru(np.zeros((16, 16, 3), dtype=np.uint8))

    assert result['gesture'] == ''
    assert result['hand_count'] == 0
    assert not result['buffering']
    assert len(recognizer.bigru_buffer) == 40
    assert recognizer.bigru_missing_hand_frames == 1

    for _ in range(recognizer.BIGRU_MISSING_HAND_GRACE_FRAMES - 1):
        result = recognizer._predict_bigru(np.zeros((16, 16, 3), dtype=np.uint8))

    assert result['hand_count'] == 0
    assert len(recognizer.bigru_buffer) == 0
    assert recognizer.bigru_missing_hand_frames == 0


def test_bigru_processes_downscaled_frame_and_predicts_after_40_frames():
    class FakeHolistic:
        def process(self, image):
            self.image_shape = image.shape
            hand = SimpleNamespace(
                landmark=[SimpleNamespace(x=0.1, y=0.2, z=0.3) for _ in range(21)]
            )
            return SimpleNamespace(
                pose_landmarks=None,
                left_hand_landmarks=hand,
                right_hand_landmarks=None
            )

    class FakeBiGruModel:
        def predict(self, sequence, verbose=0):
            assert sequence.shape == (1, 40, 225)
            self.sequence = sequence.copy()
            return np.asarray([[0.1, 0.9]], dtype=np.float32)

    holistic = FakeHolistic()
    model = FakeBiGruModel()
    recognizer = GestureRecognizer.__new__(GestureRecognizer)
    recognizer.bigru_model = model
    recognizer.bigru_holistic = holistic
    recognizer.bigru_sequence_length = 40
    recognizer.bigru_buffer = deque(
        [np.zeros(225, dtype=np.float32) for _ in range(39)],
        maxlen=40
    )
    recognizer.bigru_mean = np.zeros(225, dtype=np.float32)
    recognizer.bigru_std = np.ones(225, dtype=np.float32)
    recognizer.bigru_labels = ['HELLO', 'THANKYOU']
    recognizer.prediction_times = []

    result = recognizer._predict_bigru(np.zeros((480, 640, 3), dtype=np.uint8))

    assert holistic.image_shape == (360, 480, 3)
    assert result['gesture'] == 'THANKYOU'
    assert result['confidence'] == pytest.approx(0.9)
    assert result['buffering'] is False
    assert len(recognizer.bigru_buffer) == 40
    assert model.sequence[0, -1, 100] == pytest.approx(0.15)


def test_lstm_live_prediction_buffers_frames_then_returns_word_label():
    class FakeLstmModel:
        def predict(self, sequence, verbose=0):
            assert sequence.shape == (1, 2, 63)
            return np.asarray([[0.1, 0.9]], dtype=np.float32)

    recognizer = GestureRecognizer.__new__(GestureRecognizer)
    recognizer.last_prediction_model = 'lstm'
    recognizer.bigru_buffer = deque()
    recognizer.landmark_buffer = deque(maxlen=2)
    recognizer.sequence_length = 2
    recognizer.lstm_model = FakeLstmModel()
    recognizer.lstm_labels = ['HELLO', 'THANKYOU']
    recognizer.scaler = SimpleNamespace(n_features_in_=63)
    recognizer.prediction_times = []
    recognizer.extract_landmarks = lambda image: {
        'hand_count': 1,
        'landmarks': [[0.0] * 63]
    }
    recognizer.preprocess_landmarks = lambda landmarks: np.ones(63, dtype=np.float32)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)

    first = recognizer.predict_with_model(frame, 'lstm')
    prediction = recognizer.predict_with_model(frame, 'lstm')

    assert first['buffering'] is True
    assert first['buffered_frames'] == 1
    assert first['sequence_length'] == 2
    assert prediction['gesture'] == 'THANKYOU'
    assert prediction['gesture_type'] == 'word'
    assert prediction['confidence'] == pytest.approx(0.9)


def test_emergency_live_prediction_buffers_two_hand_landmarks_then_returns_word():
    class FakeLandmarker:
        def detect(self, image):
            self.image_data = image.numpy_view().copy()
            hand = [SimpleNamespace(x=0.1, y=0.2, z=0.3) for _ in range(21)]
            return SimpleNamespace(hand_landmarks=[hand])

    class FakeEmergencyModel:
        def predict(self, sequence, verbose=0):
            assert sequence.shape == (1, 30, 126)
            return np.asarray([[0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.1]])

    recognizer = GestureRecognizer.__new__(GestureRecognizer)
    recognizer.emergency_model = FakeEmergencyModel()
    recognizer.emergency_landmarker = FakeLandmarker()
    recognizer.emergency_sequence_length = 30
    recognizer.emergency_buffer = deque(
        [np.zeros(126, dtype=np.float32) for _ in range(29)],
        maxlen=30
    )
    recognizer.emergency_labels = [
        'ACCIDENT', 'CALL', 'DOCTOR', 'HELP', 'HOT', 'LOSE', 'PAIN', 'THIEF'
    ]
    recognizer.prediction_times = []

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[:, :16] = (10, 20, 30)
    frame[:, 16:] = (40, 50, 60)
    result = recognizer._predict_emergency(frame)

    assert result['gesture'] == 'HOT'
    assert result['confidence'] == pytest.approx(0.9)
    assert result['model_id'] == 'emergency_lstm'
    assert result['gesture_type'] == 'word'
    assert result['hand_count'] == 1
    assert result['buffering'] is False
    np.testing.assert_array_equal(
        recognizer.emergency_landmarker.image_data,
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )


def test_emergency_model_preserves_buffer_during_brief_hand_tracking_loss():
    class EmptyLandmarker:
        def detect(self, image):
            return SimpleNamespace(hand_landmarks=[])

    recognizer = GestureRecognizer.__new__(GestureRecognizer)
    recognizer.emergency_model = object()
    recognizer.emergency_landmarker = EmptyLandmarker()
    recognizer.emergency_buffer = deque(
        [np.ones(126, dtype=np.float32) for _ in range(30)],
        maxlen=30
    )
    recognizer.emergency_missing_hand_frames = 0

    result = recognizer._predict_emergency(np.zeros((16, 16, 3), dtype=np.uint8))

    assert result['gesture'] == ''
    assert result['hand_count'] == 0
    assert len(recognizer.emergency_buffer) == 30
    assert recognizer.emergency_missing_hand_frames == 1

    for _ in range(recognizer.EMERGENCY_MISSING_HAND_GRACE_FRAMES - 1):
        result = recognizer._predict_emergency(np.zeros((16, 16, 3), dtype=np.uint8))

    assert result['hand_count'] == 0
    assert len(recognizer.emergency_buffer) == 0
    assert recognizer.emergency_missing_hand_frames == 0


def test_available_models_includes_loaded_emergency_model():
    recognizer = GestureRecognizer.__new__(GestureRecognizer)
    recognizer.cnn_model_loaded = False
    recognizer.cnn_load_error = 'missing cnn'
    recognizer.bigru_model = None
    recognizer.bigru_labels = None
    recognizer.bigru_load_error = 'missing bigru'
    recognizer.emergency_model = object()
    recognizer.emergency_labels = ['HELP', 'PAIN']
    recognizer.emergency_load_error = None
    recognizer.lstm_model = None
    recognizer.lstm_labels = None
    recognizer.lstm_load_error = 'missing lstm'

    models = recognizer.get_available_models()

    assert models == [{
        'id': 'emergency_lstm',
        'name': 'Emergency Signs LSTM (2 classes)',
        'available': True,
        'selectable': True,
        'reason': ''
    }]


def test_stable_emergency_prediction_dispatches_sos(monkeypatch):
    import app as app_module

    dispatched = []
    original_sentence = app_module.current_sentence
    monkeypatch.setattr(
        app_module,
        'dispatch_emergency_sos',
        lambda **kwargs: (dispatched.append(kwargs) or {'success': True}, 200)
    )
    app_module.prediction_committer.reset()
    app_module.current_sentence = ''

    try:
        prediction = {
            'gesture': 'HELP',
            'confidence': 0.9,
            'gesture_type': 'word',
            'model_id': 'emergency_lstm'
        }

        assert app_module.commit_stable_prediction(prediction, '127.0.0.1')
        assert prediction['emergency_detected'] is True
        assert prediction['emergency_sos']['success'] is True
        assert dispatched == [{
            'source': 'Signora emergency sign recognition',
            'gesture': 'HELP',
            'client_ip': '127.0.0.1'
        }]
    finally:
        app_module.current_sentence = original_sentence
        app_module.prediction_committer.reset()


def test_low_confidence_emergency_prediction_does_not_dispatch_sos(monkeypatch):
    import app as app_module

    dispatched = []
    original_sentence = app_module.current_sentence
    monkeypatch.setattr(
        app_module,
        'dispatch_emergency_sos',
        lambda **kwargs: (dispatched.append(kwargs) or {'success': True}, 200)
    )
    app_module.prediction_committer.reset()
    app_module.current_sentence = ''

    try:
        prediction = {
            'gesture': 'HELP',
            'confidence': 0.64,
            'gesture_type': 'word',
            'model_id': 'emergency_lstm'
        }

        assert app_module.commit_stable_prediction(prediction)
        assert prediction['emergency_detected'] is True
        assert prediction['emergency_sos']['success'] is False
        assert not dispatched
    finally:
        app_module.current_sentence = original_sentence
        app_module.prediction_committer.reset()


def test_committer_blocks_held_sign_and_allows_repeat_after_release():
    committer = StablePredictionCommitter(
        cooldown_seconds=2.0,
        same_gesture_cooldown_seconds=4.0
    )

    assert committer.should_commit('A', 10.0)
    assert not committer.should_commit('A', 20.0)
    assert not committer.should_commit('B', 11.0)
    assert committer.should_commit('B', 12.0)
    committer.release()
    assert not committer.should_commit('B', 15.0)
    assert committer.should_commit('B', 16.0)


def test_prediction_api_returns_only_stable_display_labels(monkeypatch):
    import app as app_module

    class FakeRecognizer:
        def get_available_models(self):
            return [{'id': 'cnn', 'selectable': True}]

        def predict_with_model(self, image, model_id):
            return {
                'gesture': 'A',
                'confidence': 0.95,
                'hand_count': 1,
                'gesture_type': 'character',
                'top_predictions': [
                    {'label': 'A', 'confidence': 0.95},
                    {'label': 'B', 'confidence': 0.03},
                    {'label': 'C', 'confidence': 0.01}
                ]
            }

    success, encoded_frame = cv2.imencode(
        '.jpg',
        np.zeros((32, 32, 3), dtype=np.uint8)
    )
    assert success
    image_data = base64.b64encode(encoded_frame.tobytes()).decode('ascii')

    original_sentence = app_module.current_sentence
    monkeypatch.setattr(app_module, 'gesture_recognizer', FakeRecognizer())
    app_module.current_sentence = ''
    app_module.active_prediction_model = None
    app_module.prediction_smoother.reset()
    app_module.prediction_committer.reset()
    client = app_module.app.test_client()

    try:
        responses = [
            client.post(
                '/api/predict_frame',
                json={
                    'image': image_data,
                    'build_sentence': True,
                    'model_id': 'cnn'
                }
            )
            for _ in range(4)
        ]
        payloads = [response.get_json() for response in responses]

        assert all(response.status_code == 200 for response in responses)
        assert [payload['stable'] for payload in payloads] == [False, False, True, True]
        assert [payload['gesture'] for payload in payloads] == ['', '', 'A', 'A']
        assert payloads[2]['top_predictions'][0]['label'] == 'A'
        assert 'sentence' in payloads[2]
        assert 'sentence' not in payloads[3]

        monkeypatch.setattr(
            app_module.gesture_recognizer,
            'predict_with_model',
            lambda image, model_id: None
        )
        no_hand = client.post(
            '/api/predict_frame',
            json={
                'image': image_data,
                'build_sentence': True,
                'model_id': 'cnn'
            }
        ).get_json()
        assert no_hand['stable'] is False
        assert no_hand['gesture'] == ''
        assert no_hand['hand_count'] == 0
    finally:
        app_module.current_sentence = original_sentence
        app_module.current_gesture = ''
        app_module.current_confidence = 0.0
        app_module.current_hand_count = 0
        app_module.active_prediction_model = None
        app_module.prediction_smoother.reset()
        app_module.prediction_committer.reset()


@pytest.mark.parametrize(
    'model_id',
    ['cnn', 'lstm', 'swatarnik_bigru', 'emergency_lstm']
)
def test_server_camera_predicts_from_unmirrored_frame(monkeypatch, model_id):
    import app as app_module

    raw_frame = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)

    class FakeCamera:
        def isOpened(self):
            return True

        def read(self):
            return True, raw_frame.copy()

    class FakeRecognizer:
        def __init__(self):
            self.prediction_frame = None

        def predict_with_model(self, frame, requested_model_id):
            assert requested_model_id == model_id
            self.prediction_frame = frame.copy()
            return None

    previous_state = (
        app_module.gesture_recognizer,
        app_module.active_prediction_model,
        app_module.current_gesture,
        app_module.current_confidence,
        app_module.current_hand_count
    )
    recognizer = FakeRecognizer()
    monkeypatch.setattr(app_module, 'get_camera', FakeCamera)
    monkeypatch.setattr(app_module, 'gesture_recognizer', recognizer)
    app_module.active_prediction_model = None

    try:
        frames = app_module.generate_frames(model_id)
        next(frames)
        assert np.array_equal(recognizer.prediction_frame, raw_frame)
        frames.close()
    finally:
        (
            app_module.gesture_recognizer,
            app_module.active_prediction_model,
            app_module.current_gesture,
            app_module.current_confidence,
            app_module.current_hand_count
        ) = previous_state
        app_module.prediction_smoother.reset()
        app_module.prediction_committer.reset()


def test_prediction_api_exposes_sequence_model_buffer_progress(monkeypatch):
    import app as app_module

    class BufferingRecognizer:
        def get_available_models(self):
            return [{'id': 'lstm', 'selectable': True}]

        def predict_with_model(self, image, model_id):
            assert model_id == 'lstm'
            return {
                'gesture': '',
                'confidence': 0.0,
                'hand_count': 1,
                'buffering': True,
                'buffered_frames': 7,
                'sequence_length': 30,
                'top_predictions': []
            }

    success, encoded_frame = cv2.imencode(
        '.jpg',
        np.zeros((32, 32, 3), dtype=np.uint8)
    )
    assert success
    image_data = base64.b64encode(encoded_frame.tobytes()).decode('ascii')
    original_state = (
        app_module.current_gesture,
        app_module.current_confidence,
        app_module.current_hand_count,
        app_module.active_prediction_model
    )
    monkeypatch.setattr(app_module, 'gesture_recognizer', BufferingRecognizer())
    app_module.active_prediction_model = None
    app_module.prediction_smoother.reset()
    app_module.prediction_committer.reset()

    try:
        response = app_module.app.test_client().post(
            '/api/predict_frame',
            json={'image': image_data, 'model_id': 'lstm'}
        )

        assert response.status_code == 200
        assert response.get_json()['model_buffering'] is True
        assert response.get_json()['buffered_frames'] == 7
        assert response.get_json()['sequence_length'] == 30
    finally:
        (
            app_module.current_gesture,
            app_module.current_confidence,
            app_module.current_hand_count,
            app_module.active_prediction_model
        ) = original_state
        app_module.prediction_smoother.reset()
        app_module.prediction_committer.reset()


def test_prediction_api_exposes_unconfirmed_candidate(monkeypatch):
    import app as app_module

    class LowConfidenceRecognizer:
        def get_available_models(self):
            return [{'id': 'swatarnik_bigru', 'selectable': True}]

        def predict_with_model(self, image, model_id):
            return {
                'gesture': 'HELLO',
                'confidence': 0.42,
                'hand_count': 1,
                'gesture_type': 'word',
                'top_predictions': [{'label': 'HELLO', 'confidence': 0.42}]
            }

    success, encoded_frame = cv2.imencode(
        '.jpg',
        np.zeros((32, 32, 3), dtype=np.uint8)
    )
    assert success
    image_data = base64.b64encode(encoded_frame.tobytes()).decode('ascii')
    original_state = (
        app_module.current_gesture,
        app_module.current_confidence,
        app_module.current_hand_count,
        app_module.active_prediction_model
    )
    monkeypatch.setattr(app_module, 'gesture_recognizer', LowConfidenceRecognizer())
    app_module.active_prediction_model = None
    app_module.prediction_smoother.reset()
    app_module.prediction_committer.reset()

    try:
        response = app_module.app.test_client().post(
            '/api/predict_frame',
            json={'image': image_data, 'model_id': 'swatarnik_bigru'}
        )
        payload = response.get_json()

        assert response.status_code == 200
        assert payload['stable'] is False
        assert payload['gesture'] == ''
        assert payload['candidate_gesture'] == 'HELLO'
        assert payload['candidate_confidence'] == pytest.approx(0.42)
    finally:
        (
            app_module.current_gesture,
            app_module.current_confidence,
            app_module.current_hand_count,
            app_module.active_prediction_model
        ) = original_state
        app_module.prediction_smoother.reset()
        app_module.prediction_committer.reset()


def test_cnn_raises_for_malformed_model_output():
    class InvalidOutputModel:
        def predict(self, features, verbose=0):
            return np.asarray([[0.2, 0.8]], dtype=np.float32)

    recognizer = GestureRecognizer.__new__(GestureRecognizer)
    recognizer.last_prediction_model = 'cnn'
    recognizer.bigru_buffer = deque()
    recognizer.landmark_buffer = deque()
    recognizer.model = InvalidOutputModel()
    recognizer.class_labels = ['A']
    recognizer.prediction_times = []
    recognizer.extract_landmarks = lambda image: {'hand_count': 1, 'landmarks': []}
    recognizer.preprocess_landmarks = lambda landmarks: np.zeros(63, dtype=np.float32)

    with pytest.raises(ValueError, match='CNN prediction output'):
        recognizer.predict_with_model(
            np.zeros((16, 16, 3), dtype=np.uint8),
            'cnn'
        )


def test_cnn_keeps_highest_scoring_class_when_letter_score_is_close():
    class FakeCnnModel:
        def predict(self, features, verbose=0):
            assert features.shape == (1, 63)
            return np.asarray([[0.44, 0.56]], dtype=np.float32)

    recognizer = GestureRecognizer.__new__(GestureRecognizer)
    recognizer.last_prediction_model = 'cnn'
    recognizer.bigru_buffer = deque()
    recognizer.landmark_buffer = deque()
    recognizer.model = FakeCnnModel()
    recognizer.class_labels = ['A', 'HELP']
    recognizer.prediction_times = []
    recognizer.extract_landmarks = lambda image: {
        'hand_count': 1,
        'landmarks': [[0.0] * 63]
    }
    recognizer.preprocess_landmarks = lambda landmarks: np.zeros(63, dtype=np.float32)

    prediction = recognizer.predict_with_model(
        np.zeros((16, 16, 3), dtype=np.uint8),
        'cnn'
    )

    assert prediction['gesture'] == 'HELP'
    assert prediction['confidence'] == pytest.approx(0.56)
    assert prediction['top_predictions'][0]['label'] == prediction['gesture']
    assert prediction['top_predictions'][0]['confidence'] == prediction['confidence']


def test_prediction_api_reports_inference_errors_and_resets_state(monkeypatch):
    import app as app_module

    class FailingRecognizer:
        def __init__(self):
            self.reset_called = False

        def get_available_models(self):
            return [{'id': 'cnn', 'selectable': True}]

        def predict_with_model(self, image, model_id):
            raise RuntimeError('private model internals')

        def reset_temporal_state(self):
            self.reset_called = True

    success, encoded_frame = cv2.imencode(
        '.jpg',
        np.zeros((32, 32, 3), dtype=np.uint8)
    )
    assert success
    image_data = base64.b64encode(encoded_frame.tobytes()).decode('ascii')

    original_state = (
        app_module.current_sentence,
        app_module.current_gesture,
        app_module.current_confidence,
        app_module.current_hand_count,
        app_module.active_prediction_model
    )
    failing_recognizer = FailingRecognizer()
    monkeypatch.setattr(app_module, 'gesture_recognizer', failing_recognizer)
    app_module.current_gesture = 'STALE'
    app_module.current_confidence = 0.99
    app_module.current_hand_count = 1
    app_module.active_prediction_model = None
    app_module.prediction_smoother.reset()
    app_module.prediction_committer.reset()

    try:
        response = app_module.app.test_client().post(
            '/api/predict_frame',
            json={'image': image_data, 'model_id': 'cnn'}
        )

        assert response.status_code == 500
        assert response.is_json
        assert 'private model internals' not in response.get_json()['error']
        assert app_module.current_gesture == ''
        assert app_module.current_confidence == 0.0
        assert app_module.current_hand_count == 0
        assert failing_recognizer.reset_called
    finally:
        (
            app_module.current_sentence,
            app_module.current_gesture,
            app_module.current_confidence,
            app_module.current_hand_count,
            app_module.active_prediction_model
        ) = original_state
        app_module.prediction_smoother.reset()
        app_module.prediction_committer.reset()


@pytest.mark.parametrize(
    ('payload', 'expected_status'),
    [
        ('not json', 400),
        ({'image': 'not-base64!!!', 'model_id': 'cnn'}, 400)
    ]
)
def test_prediction_api_rejects_malformed_requests(payload, expected_status):
    import app as app_module

    client = app_module.app.test_client()
    if isinstance(payload, str):
        response = client.post(
            '/api/predict_frame',
            data=payload,
            content_type='application/json'
        )
    else:
        response = client.post('/api/predict_frame', json=payload)

    assert response.status_code == expected_status
    assert response.is_json
    assert response.get_json()['error']
