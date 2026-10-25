"""
Gesture Recognition Module for ISL System
Uses MediaPipe for hand tracking and deep learning models for classification
"""
import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import time
import json
import io
import zipfile
import h5py
import logging
from collections import deque

logger = logging.getLogger(__name__)

class GestureRecognizer:
    """Main class for ISL gesture recognition"""

    BIGRU_MISSING_HAND_GRACE_FRAMES = 8
    EMERGENCY_MISSING_HAND_GRACE_FRAMES = 8
    
    def __init__(self, model_path: str = None, config: Dict = None):
        """
        Initialize the gesture recognizer
        
        Args:
            model_path: Path to trained model
            config: Configuration dictionary
        """
        self.config = config or {}
        self.model_path = model_path
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hand tracking
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.get('MAX_NUM_HANDS', 2),
            min_detection_confidence=self.config.get('HAND_DETECTION_CONFIDENCE', 0.7),
            min_tracking_confidence=self.config.get('HAND_TRACKING_CONFIDENCE', 0.5)
        )
        
        # Model variables
        self.model = None
        self.class_labels = None
        self.scaler = None
        self.cnn_model_loaded = False
        self.cnn_load_error = None

        # LSTM variables
        self.lstm_model = None
        self.lstm_labels = None
        self.sequence_length = 30
        self.landmark_buffer = deque(maxlen=30)
        self.bigru_model = None
        self.bigru_labels = None
        self.bigru_mean = None
        self.bigru_std = None
        self.bigru_sequence_length = 40
        self.bigru_buffer = deque(maxlen=self.bigru_sequence_length)
        self.bigru_missing_hand_frames = 0
        self.bigru_holistic = None
        self.bigru_load_error = None
        self.emergency_model = None
        self.emergency_labels = None
        self.emergency_sequence_length = 30
        self.emergency_buffer = deque(maxlen=self.emergency_sequence_length)
        self.emergency_missing_hand_frames = 0
        self.emergency_landmarker = None
        self.emergency_load_error = None
        self.last_prediction_model = 'cnn'
        
        # Performance tracking
        self.prediction_times = []
        self.frame_buffer = []
        self.max_buffer_size = 10
        
        # Load model if path provided
        if self.model_path and os.path.exists(self.model_path):
            self.load_model()
        else:
            self.cnn_load_error = f'Model file was not found: {self.model_path}'
            print("⚠️ Model not found. Creating placeholder model...")
            self.create_placeholder_model()
        
        # Try to load LSTM model
        self._load_lstm_model()
        self._load_bigru_model()
        self._load_emergency_model()
    
    def create_placeholder_model(self):
        """Create a placeholder model for testing purposes"""
        # Create a simple CNN model structure for demonstration
        self.model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(63,)),  # 21 landmarks * 3 coordinates
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(36, activation='softmax')  # 26 letters + 10 numbers
        ])
        
        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create class labels
        self.class_labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + \
                           [str(i) for i in range(10)]
        
        print("📝 Placeholder model created with {} classes".format(len(self.class_labels)))
    
    def load_model(self):
        """Load trained model from file"""
        try:
            self.model = keras.models.load_model(self.model_path)
            self.cnn_model_loaded = True
            print(f"✅ Model loaded from {self.model_path}")
            
            # Load class labels if available
            label_path = self.model_path.replace('.h5', '_labels.pkl')
            if os.path.exists(label_path):
                with open(label_path, 'rb') as f:
                    label_encoder = pickle.load(f)
                    # Handle both LabelEncoder object and list of classes
                    if hasattr(label_encoder, 'classes_'):
                        self.class_labels = label_encoder.classes_.tolist()
                    else:
                        self.class_labels = label_encoder
            
            # Load scaler if available
            scaler_path = self.model_path.replace('.h5', '_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)

            if tuple(self.model.input_shape[1:]) != (63,):
                raise ValueError(
                    f'Expected CNN input shape (63,), got {self.model.input_shape[1:]}.'
                )
            if not self.class_labels:
                raise ValueError('CNN class labels are missing.')
            if int(self.model.output_shape[-1]) != len(self.class_labels):
                raise ValueError(
                    f'CNN class label count ({len(self.class_labels)}) does not match '
                    f'model outputs ({self.model.output_shape[-1]}).'
                )
            if self.scaler is None or getattr(self.scaler, 'n_features_in_', None) != 63:
                raise ValueError('CNN feature scaler is missing or does not expect 63 features.')
                    
        except Exception as e:
            self.cnn_model_loaded = False
            self.cnn_load_error = str(e)
            print(f"❌ Error loading model: {e}")
            self.create_placeholder_model()
    
    def extract_landmarks(self, image: np.ndarray) -> Optional[Dict]:
        """
        Extract hand landmarks from image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary containing landmarks or None
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        
        # Process the image
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            landmarks_data = {
                'landmarks': [],
                'handedness': [],
                'bbox': [],
                'hand_count': len(results.multi_hand_landmarks)
            }
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Extract landmark coordinates
                landmark_list = []
                x_coords = []
                y_coords = []
                
                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    z = landmark.z
                    
                    landmark_list.extend([x, y, z])
                    x_coords.append(x)
                    y_coords.append(y)
                
                landmarks_data['landmarks'].append(landmark_list)
                
                # Calculate bounding box
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                landmarks_data['bbox'].append([min_x, min_y, max_x, max_y])
                
                # Get handedness (left/right)
                if results.multi_handedness:
                    handedness = results.multi_handedness[idx].classification[0].label
                    landmarks_data['handedness'].append(handedness)
            
            return landmarks_data
        
        return None
    
    def preprocess_landmarks(self, landmarks_data: Dict) -> np.ndarray:
        """
        Preprocess landmarks for model input

        Args:
            landmarks_data: Raw landmarks data

        Returns:
            Preprocessed feature vector
        """
        if not landmarks_data or not landmarks_data['landmarks']:
            return np.zeros((63,))  # 21 landmarks * 3 coordinates

        # Process both hands if available, use the one with better visibility
        best_features = None
        best_score = -1

        for hand_idx, landmarks in enumerate(landmarks_data['landmarks']):
            # Convert to numpy array
            features = np.array(landmarks, dtype=np.float32)

            # Calculate a simple score based on landmark spread (better hand detection)
            wrist = features[:3]  # First 3 values are wrist x,y,z
            distances = np.linalg.norm(features.reshape(-1, 3) - wrist.reshape(1, 3), axis=1)
            score = np.mean(distances)  # Higher score = better hand pose

            # Normalize landmarks
            features = self.normalize_landmarks(features)

            # Apply scaler if available
            if self.scaler:
                features = self.scaler.transform(features.reshape(1, -1)).flatten()

            # Keep the hand with the best score
            if score > best_score:
                best_score = score
                best_features = features

        return best_features if best_features is not None else np.zeros((63,))
    
    def _buffer_landmarks(self, features):
        """Add normalized landmarks to the sequence buffer for LSTM"""
        self.landmark_buffer.append(features.copy())
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmark coordinates
        
        Args:
            landmarks: Raw landmark coordinates
            
        Returns:
            Normalized landmarks
        """
        # Reshape to (21, 3) for processing
        landmarks_2d = landmarks.reshape(-1, 3)
        
        # Get wrist landmark (index 0) as reference point
        wrist = landmarks_2d[0]
        
        # Translate to origin (wrist = [0, 0, 0])
        landmarks_2d = landmarks_2d - wrist
        
        # Scale to unit size
        max_distance = np.max(np.linalg.norm(landmarks_2d, axis=1))
        if max_distance > 0:
            landmarks_2d = landmarks_2d / max_distance
        
        return landmarks_2d.flatten()
    
    def predict(self, image: np.ndarray) -> Optional[Dict]:
        """
        Predict gesture from image
        
        Args:
            image: Input image
            
        Returns:
            Prediction results or None
        """
        return self.predict_with_model(image, 'cnn')

    def predict_with_model(self, image: np.ndarray, model_id: str = 'cnn') -> Optional[Dict]:
        """Run prediction with a selected, connected model."""
        if model_id != self.last_prediction_model:
            self.reset_temporal_state()
            self.last_prediction_model = model_id

        try:
            if model_id == 'swatarnik_bigru':
                return self._predict_bigru(image)
            if model_id == 'emergency_lstm':
                return self._predict_emergency(image)
            if model_id == 'lstm':
                return self._predict_lstm(image)
            if model_id != 'cnn':
                raise ValueError(f"Model '{model_id}' is not connected for live sign-to-text.")
            return self._predict_cnn(image)
        except Exception:
            self.reset_temporal_state()
            raise

    def _predict_cnn(self, image: np.ndarray) -> Optional[Dict]:
        """Run one static-frame CNN inference and validate its class scores."""
        start_time = time.time()
        if image is None or not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Prediction input must be a 3-channel BGR image.')

        landmarks_data = self.extract_landmarks(image)
        if landmarks_data is None:
            return None

        features = self.preprocess_landmarks(landmarks_data)
        if features.shape != (63,) or not np.all(np.isfinite(features)):
            raise ValueError('CNN preprocessing must return 63 finite landmark features.')
        if self.model is None:
            raise RuntimeError('CNN model is not loaded.')

        self._buffer_landmarks(features)
        predictions = np.asarray(
            self.model.predict(features.reshape(1, -1), verbose=0)
        )
        if (
            predictions.ndim != 2
            or predictions.shape != (1, len(self.class_labels))
            or not np.all(np.isfinite(predictions))
        ):
            raise ValueError(
                'CNN prediction output must contain one finite score per configured class.'
            )

        predicted_class_idx = int(np.argmax(predictions[0]))
        gesture = self.class_labels[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])

        top_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = [
            {
                'label': self.class_labels[int(index)],
                'confidence': float(predictions[0][index])
            }
            for index in top_indices
        ]

        logger.debug(
            'CNN top-three=%s final=%s score=%.4f',
            top_predictions,
            gesture,
            confidence
        )

        prediction_time = time.time() - start_time
        self.prediction_times.append(prediction_time)
        if len(self.prediction_times) > 100:
            self.prediction_times.pop(0)

        return {
            'gesture': gesture,
            'confidence': confidence,
            'model_id': 'cnn',
            'gesture_type': 'character',
            'top_predictions': top_predictions,
            'landmarks_data': landmarks_data,
            'prediction_time': prediction_time,
            'hand_count': landmarks_data['hand_count'],
        }

    def _predict_lstm(self, image: np.ndarray) -> Optional[Dict]:
        """Predict a dynamic gesture from a rolling sequence of hand landmarks."""
        if self.lstm_model is None or not self.lstm_labels:
            raise RuntimeError('LSTM model is not available: model or labels were not loaded.')
        if self.scaler is None or getattr(self.scaler, 'n_features_in_', None) != 63:
            raise RuntimeError(
                'LSTM inference requires the 63-feature scaler used by sequence collection.'
            )
        if image is None or not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Prediction input must be a 3-channel BGR image.')

        started_at = time.time()
        landmarks_data = self.extract_landmarks(image)
        if landmarks_data is None:
            self.landmark_buffer.clear()
            return {
                'gesture': '',
                'confidence': 0.0,
                'model_id': 'lstm',
                'gesture_type': 'word',
                'hand_count': 0,
                'buffering': False,
                'top_predictions': []
            }

        features = self.preprocess_landmarks(landmarks_data)
        if features.shape != (63,) or not np.all(np.isfinite(features)):
            raise ValueError('LSTM preprocessing must return 63 finite landmark features.')
        self.landmark_buffer.append(features.copy())
        if len(self.landmark_buffer) < self.sequence_length:
            return {
                'gesture': '',
                'confidence': 0.0,
                'model_id': 'lstm',
                'gesture_type': 'word',
                'hand_count': int(landmarks_data.get('hand_count', 1)),
                'buffering': True,
                'buffered_frames': len(self.landmark_buffer),
                'sequence_length': self.sequence_length,
                'top_predictions': []
            }

        sequence = np.asarray(self.landmark_buffer, dtype=np.float32)
        predictions = np.asarray(
            self.lstm_model.predict(
                sequence.reshape(1, self.sequence_length, 63),
                verbose=0
            )
        )
        if (
            predictions.ndim != 2
            or predictions.shape != (1, len(self.lstm_labels))
            or not np.all(np.isfinite(predictions))
        ):
            raise ValueError(
                'LSTM prediction output must contain one finite score per configured class.'
            )

        scores = predictions[0]
        class_index = int(np.argmax(scores))
        top_indices = np.argsort(scores)[-3:][::-1]
        top_predictions = [
            {
                'label': self.lstm_labels[int(index)],
                'confidence': float(scores[index])
            }
            for index in top_indices
        ]
        elapsed = time.time() - started_at
        self.prediction_times.append(elapsed)
        if len(self.prediction_times) > 100:
            self.prediction_times.pop(0)
        return {
            'gesture': self.lstm_labels[class_index],
            'confidence': float(scores[class_index]),
            'model_id': 'lstm',
            'gesture_type': 'word',
            'top_predictions': top_predictions,
            'prediction_time': elapsed,
            'hand_count': int(landmarks_data.get('hand_count', 1)),
            'buffering': False
        }

    def _load_bigru_model(self):
        """Load the Swatarnik BiGRU model and its inference metadata."""
        model_dir = os.path.join('models', 'Swatarnik Dataset')
        model_path = os.path.join(model_dir, 'best_isl_bigru.keras')
        labels_path = os.path.join(model_dir, 'classes.json')
        mean_path = os.path.join(model_dir, 'mean.npy')
        std_path = os.path.join(model_dir, 'std.npy')
        required_paths = (model_path, labels_path, mean_path, std_path)

        if not all(os.path.exists(path) for path in required_paths):
            missing = [os.path.basename(path) for path in required_paths if not os.path.exists(path)]
            self.bigru_load_error = f"Missing required files: {', '.join(missing)}."
            print(f"ℹ️ Swatarnik BiGRU is unavailable: {self.bigru_load_error}")
            return

        try:
            try:
                model = keras.models.load_model(model_path)
            except (TypeError, ValueError) as load_error:
                print(f"ℹ️ Trying legacy loader for Keras 3 checkpoint: {load_error}")
                model = self._load_keras3_recurrent_checkpoint(model_path)
            input_shape = tuple(model.input_shape[1:])
            if input_shape != (40, 225):
                raise ValueError(f"Expected model input (40, 225), got {input_shape}")

            with open(labels_path, encoding='utf-8') as label_file:
                label_data = json.load(label_file)
            if isinstance(label_data, dict):
                labels = [label_data[str(index)] for index in range(len(label_data))]
            else:
                labels = label_data
            if len(labels) != int(model.output_shape[-1]):
                raise ValueError(
                    f"Class label count ({len(labels)}) does not match model outputs "
                    f"({model.output_shape[-1]})."
                )

            mean = np.load(mean_path).reshape(-1)
            std = np.load(std_path).reshape(-1)
            if mean.size != 225 or std.size != 225:
                raise ValueError('BiGRU normalization statistics must each contain 225 values.')
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
                raise ValueError('BiGRU normalization statistics contain non-finite values.')

            self.bigru_model = model
            self.bigru_labels = [str(label).upper() for label in labels]
            self.bigru_mean = mean
            self.bigru_std = np.where(np.abs(std) < 1e-6, 1.0, std)
            self.bigru_holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True,
                refine_face_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print(f"✅ Swatarnik BiGRU loaded ({len(self.bigru_labels)} classes)")
        except Exception as e:
            self.bigru_load_error = str(e)
            self.bigru_model = None
            self.bigru_labels = None
            self.bigru_mean = None
            self.bigru_std = None
            if self.bigru_holistic is not None:
                self.bigru_holistic.close()
                self.bigru_holistic = None
            print(f"❌ Could not load Swatarnik BiGRU: {e}")

    def _load_emergency_model(self):
        """Load the emergency-sign LSTM and its MediaPipe hand detector."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        model_dir = os.path.join(project_root, 'models', 'Emergency Signs')
        model_path = os.path.join(model_dir, 'isl_emergency_lstm.keras')
        labels_path = os.path.join(model_dir, 'class_labels.json')
        task_path = os.path.join(model_dir, 'hand_landmarker.task')
        required_paths = (model_path, labels_path, task_path)

        if not all(os.path.exists(path) for path in required_paths):
            missing = [os.path.basename(path) for path in required_paths if not os.path.exists(path)]
            self.emergency_load_error = f"Missing required files: {', '.join(missing)}."
            print(f"ℹ️ Emergency Signs model is unavailable: {self.emergency_load_error}")
            return

        try:
            try:
                model = keras.models.load_model(model_path)
            except (TypeError, ValueError) as load_error:
                print(f"ℹ️ Trying compatibility loader for Emergency LSTM: {load_error}")
                model = self._load_keras3_recurrent_checkpoint(model_path)
            with open(labels_path, encoding='utf-8') as label_file:
                labels = json.load(label_file)
            if not isinstance(labels, list) or not labels:
                raise ValueError('Emergency class labels must be a non-empty JSON list.')
            if tuple(model.input_shape[1:]) != (self.emergency_sequence_length, 126):
                raise ValueError(
                    f'Expected Emergency Signs input '
                    f'({self.emergency_sequence_length}, 126), got {model.input_shape[1:]}.'
                )
            if int(model.output_shape[-1]) != len(labels):
                raise ValueError(
                    f'Emergency class label count ({len(labels)}) does not match model '
                    f'outputs ({model.output_shape[-1]}).'
                )

            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            with open(task_path, 'rb') as task_file:
                task_buffer = task_file.read()
            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_buffer=task_buffer
                ),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5
            )
            self.emergency_landmarker = vision.HandLandmarker.create_from_options(options)
            self.emergency_model = model
            self.emergency_labels = [str(label).upper() for label in labels]
            self.emergency_load_error = None
            print(f"✅ Emergency Signs LSTM loaded ({len(self.emergency_labels)} classes)")
        except Exception as error:
            self.emergency_load_error = str(error)
            self.emergency_model = None
            self.emergency_labels = None
            if self.emergency_landmarker is not None:
                self.emergency_landmarker.close()
                self.emergency_landmarker = None
            print(f"❌ Could not load Emergency Signs LSTM: {error}")

    def _predict_emergency(self, image: np.ndarray) -> Optional[Dict]:
        """Predict an emergency sign from a rolling 30-frame, two-hand sequence."""
        if self.emergency_model is None or self.emergency_landmarker is None:
            raise RuntimeError(
                'Emergency Signs model is not available: '
                f'{self.emergency_load_error or "model was not loaded"}'
            )
        if image is None or not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Prediction input must be a 3-channel BGR image.')

        started_at = time.time()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        from mediapipe import Image, ImageFormat

        detection = self.emergency_landmarker.detect(
            Image(image_format=ImageFormat.SRGB, data=rgb_image)
        )
        hand_landmarks = detection.hand_landmarks
        if not hand_landmarks:
            self.emergency_missing_hand_frames += 1
            if self.emergency_missing_hand_frames >= self.EMERGENCY_MISSING_HAND_GRACE_FRAMES:
                self.emergency_buffer.clear()
                self.emergency_missing_hand_frames = 0
            return {
                'gesture': '',
                'confidence': 0.0,
                'model_id': 'emergency_lstm',
                'gesture_type': 'word',
                'hand_count': 0,
                'buffering': False,
                'top_predictions': []
            }

        self.emergency_missing_hand_frames = 0
        features = np.zeros((2, 21, 3), dtype=np.float32)
        for hand_index, landmarks in enumerate(hand_landmarks[:2]):
            features[hand_index] = np.asarray(
                [[point.x, point.y, point.z] for point in landmarks],
                dtype=np.float32
            )
        self.emergency_buffer.append(features.reshape(-1))
        hand_count = min(len(hand_landmarks), 2)
        if len(self.emergency_buffer) < self.emergency_sequence_length:
            return {
                'gesture': '',
                'confidence': 0.0,
                'model_id': 'emergency_lstm',
                'gesture_type': 'word',
                'hand_count': hand_count,
                'buffering': True,
                'buffered_frames': len(self.emergency_buffer),
                'sequence_length': self.emergency_sequence_length,
                'top_predictions': []
            }

        sequence = np.asarray(self.emergency_buffer, dtype=np.float32)
        predictions = np.asarray(
            self.emergency_model.predict(
                sequence.reshape(1, self.emergency_sequence_length, 126),
                verbose=0
            )
        )
        if (
            predictions.ndim != 2
            or predictions.shape != (1, len(self.emergency_labels))
            or not np.all(np.isfinite(predictions))
        ):
            raise ValueError(
                'Emergency Signs prediction output must contain one finite score per label.'
            )

        scores = predictions[0]
        class_index = int(np.argmax(scores))
        top_indices = np.argsort(scores)[-3:][::-1]
        top_predictions = [
            {
                'label': self.emergency_labels[int(index)],
                'confidence': float(scores[index])
            }
            for index in top_indices
        ]
        elapsed = time.time() - started_at
        self.prediction_times.append(elapsed)
        if len(self.prediction_times) > 100:
            self.prediction_times.pop(0)
        return {
            'gesture': self.emergency_labels[class_index],
            'confidence': float(scores[class_index]),
            'model_id': 'emergency_lstm',
            'gesture_type': 'word',
            'top_predictions': top_predictions,
            'prediction_time': elapsed,
            'hand_count': hand_count,
            'buffering': False
        }

    def _load_keras3_recurrent_checkpoint(self, model_path: str):
        """Load Keras 3 recurrent weights into the app's TensorFlow/Keras model."""
        with zipfile.ZipFile(model_path) as archive:
            model_config = json.loads(archive.read('config.json'))
            weights_data = archive.read('model.weights.h5')

        def legacy_config(value):
            if isinstance(value, list):
                return [legacy_config(item) for item in value]
            if not isinstance(value, dict):
                return value

            converted = {
                key: legacy_config(item)
                for key, item in value.items()
                if key not in {'shared_object_id', 'quantization_config'}
            }
            if converted.get('class_name') == 'DTypePolicy':
                return converted.get('config', {}).get('name', 'float32')
            layer_config = converted.get('config')
            if isinstance(layer_config, dict) and 'batch_shape' in layer_config:
                layer_config['batch_input_shape'] = layer_config.pop('batch_shape')
            if converted.get('class_name') == 'InputLayer' and isinstance(layer_config, dict):
                layer_config.pop('optional', None)
            if converted.get('class_name') in {'LSTM', 'GRU', 'SimpleRNN'}:
                if isinstance(layer_config, dict):
                    layer_config.pop('seed', None)
            return converted

        model = keras.models.model_from_json(json.dumps(legacy_config(model_config)))
        with h5py.File(io.BytesIO(weights_data), 'r') as weight_file:
            layer_groups = weight_file['layers']
            recurrent_groups = sorted(
                (name for name in layer_groups if name == 'bidirectional' or name.startswith('bidirectional_')),
                key=lambda name: int(name.rsplit('_', 1)[1]) if '_' in name else 0
            )
            dense_groups = sorted(
                (name for name in layer_groups if name == 'dense' or name.startswith('dense_')),
                key=lambda name: int(name.rsplit('_', 1)[1]) if '_' in name else 0
            )
            recurrent_index = 0
            dense_index = 0
            for layer in model.layers:
                if isinstance(layer, keras.layers.Bidirectional):
                    if recurrent_index >= len(recurrent_groups):
                        raise ValueError('Keras 3 checkpoint is missing bidirectional layer weights.')
                    checkpoint_name = recurrent_groups[recurrent_index]
                    recurrent_index += 1
                    for direction, recurrent_layer in (
                        ('forward_layer', layer.forward_layer),
                        ('backward_layer', layer.backward_layer)
                    ):
                        group_path = (
                            f'layers/{checkpoint_name}/{direction}/cell/vars'
                        )
                        if group_path not in weight_file:
                            raise ValueError(
                                f'Keras 3 checkpoint is missing weights for '
                                f'{checkpoint_name}.{direction}.'
                            )
                        group = weight_file[group_path]
                        recurrent_layer.set_weights([
                            group[str(index)][()]
                            for index in range(len(group))
                        ])
                elif isinstance(layer, keras.layers.Dense):
                    if dense_index >= len(dense_groups):
                        raise ValueError('Keras 3 checkpoint is missing dense layer weights.')
                    checkpoint_name = dense_groups[dense_index]
                    dense_index += 1
                    group_path = f'layers/{checkpoint_name}/vars'
                    if group_path not in weight_file:
                        raise ValueError(
                            f'Keras 3 checkpoint is missing weights for {checkpoint_name}.'
                        )
                    group = weight_file[group_path]
                    layer.set_weights([
                        group[str(index)][()]
                        for index in range(len(group))
                    ])
                elif layer.weights:
                    raise ValueError(
                        f'Keras 3 compatibility loader does not support '
                        f'weighted layer {layer.name} ({type(layer).__name__}).'
                    )
        return model

    def _predict_bigru(self, image: np.ndarray) -> Optional[Dict]:
        """Predict an ISL word from a rolling 40-frame Holistic landmark sequence."""
        if self.bigru_model is None or self.bigru_holistic is None:
            raise RuntimeError(
                'Swatarnik BiGRU is not available: '
                f'{self.bigru_load_error or "model was not loaded"}'
            )
        if image is None or not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Prediction input must be a 3-channel BGR image.')

        started_at = time.time()
        frame_width = image.shape[1]
        if frame_width > 480:
            frame_height = int(image.shape[0] * 480 / frame_width)
            image = cv2.resize(
                image,
                (480, frame_height),
                interpolation=cv2.INTER_AREA
            )
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.bigru_holistic.process(rgb_image)

        def landmarks_or_zeros(landmarks, dimensions):
            if not landmarks:
                return np.zeros(dimensions, dtype=np.float32)
            values = [[point.x, point.y, point.z] for point in landmarks.landmark]
            return np.asarray(values, dtype=np.float32).reshape(-1)

        # The saved model and bundled statistics use 225 XYZ values: pose (99),
        # left hand (63), and right hand (63); visibility is not part of its input.
        features = np.concatenate([
            landmarks_or_zeros(results.pose_landmarks, 99),
            landmarks_or_zeros(results.left_hand_landmarks, 63),
            landmarks_or_zeros(results.right_hand_landmarks, 63)
        ])
        if features.size != 225:
            raise ValueError(f'Expected 225 Holistic features, received {features.size}.')
        features[1::3] *= image.shape[0] / image.shape[1]
        hand_count = int(
            bool(results.left_hand_landmarks) + bool(results.right_hand_landmarks)
        )
        if not hand_count:
            self.bigru_missing_hand_frames += 1
            if self.bigru_missing_hand_frames >= self.BIGRU_MISSING_HAND_GRACE_FRAMES:
                self.bigru_buffer.clear()
                self.bigru_missing_hand_frames = 0
            return {
                'gesture': '',
                'confidence': 0.0,
                'model_id': 'swatarnik_bigru',
                'gesture_type': 'word',
                'hand_count': 0,
                'buffering': False,
                'top_predictions': []
            }

        self.bigru_missing_hand_frames = 0
        self.bigru_buffer.append(features)

        if len(self.bigru_buffer) < self.bigru_sequence_length:
            return {
                'gesture': '',
                'confidence': 0.0,
                'model_id': 'swatarnik_bigru',
                'gesture_type': 'word',
                'hand_count': hand_count,
                'buffering': True,
                'buffered_frames': len(self.bigru_buffer),
                'sequence_length': self.bigru_sequence_length,
                'top_predictions': []
            }

        sequence = np.asarray(self.bigru_buffer, dtype=np.float32)
        sequence = (sequence - self.bigru_mean) / self.bigru_std
        raw_prediction = np.asarray(
            self.bigru_model.predict(sequence.reshape(1, 40, 225), verbose=0)
        )
        if (
            raw_prediction.ndim != 2
            or raw_prediction.shape != (1, len(self.bigru_labels))
            or not np.all(np.isfinite(raw_prediction))
        ):
            raise ValueError(
                'BiGRU prediction output must contain one finite score per configured class.'
            )
        prediction = raw_prediction[0]
        class_index = int(np.argmax(prediction))
        if class_index >= len(self.bigru_labels):
            raise ValueError(f'BiGRU returned an invalid class index: {class_index}.')

        confidence = float(prediction[class_index])
        top_indices = np.argsort(prediction)[-3:][::-1]
        top_predictions = [
            {
                'label': self.bigru_labels[int(index)],
                'confidence': float(prediction[index])
            }
            for index in top_indices
        ]
        gesture = self.bigru_labels[class_index]
        logger.debug(
            'Swatarnik BiGRU top-three=%s final=%s score=%.4f',
            top_predictions,
            gesture,
            confidence
        )
        elapsed = time.time() - started_at
        self.prediction_times.append(elapsed)
        if len(self.prediction_times) > 100:
            self.prediction_times.pop(0)
        return {
            'gesture': gesture,
            'confidence': confidence,
            'model_id': 'swatarnik_bigru',
            'gesture_type': 'word',
            'top_predictions': top_predictions,
            'prediction_time': elapsed,
            'hand_count': hand_count,
            'buffering': False,
        }

    def get_model_status(self) -> List[Dict]:
        """Report readiness and load errors for every configured model."""
        return [
            {
                'id': 'cnn',
                'name': 'ISL Landmark CNN (61 classes)',
                'available': self.cnn_model_loaded,
                'selectable': self.cnn_model_loaded,
                'reason': self.cnn_load_error or ''
            },
            {
                'id': 'swatarnik_bigru',
                'name': f'Swatarnik BiGRU ({len(self.bigru_labels or [])} classes)',
                'available': self.bigru_model is not None,
                'selectable': self.bigru_model is not None,
                'reason': self.bigru_load_error or ''
            },
            {
                'id': 'emergency_lstm',
                'name': f'Emergency Signs LSTM ({len(self.emergency_labels or [])} classes)',
                'available': self.emergency_model is not None,
                'selectable': self.emergency_model is not None,
                'reason': self.emergency_load_error or ''
            },
            {
                'id': 'lstm',
                'name': f'Dynamic LSTM ({len(self.lstm_labels or [])} classes)',
                'available': self.lstm_model is not None and bool(self.lstm_labels),
                'selectable': self.lstm_model is not None and bool(self.lstm_labels),
                'reason': self.lstm_load_error or ''
            },
        ]

    def get_available_models(self) -> List[Dict]:
        """Report connected models that are ready for live sign-to-text."""
        return [model for model in self.get_model_status() if model['selectable']]

    def reset_temporal_state(self) -> None:
        """Clear rolling model inputs after an inference failure."""
        self.landmark_buffer.clear()
        self.bigru_buffer.clear()
        self.bigru_missing_hand_frames = 0
        if hasattr(self, 'emergency_buffer'):
            self.emergency_buffer.clear()
            self.emergency_missing_hand_frames = 0
    
    def _load_lstm_model(self):
        """Load the dynamic LSTM model and validate its inference metadata."""
        lstm_path = os.path.join('models', 'trained', 'lstm_model.h5')
        lstm_labels_path = os.path.join('models', 'trained', 'lstm_model_labels.pkl')
        self.lstm_load_error = None
        
        if os.path.exists(lstm_path):
            try:
                model = keras.models.load_model(lstm_path)
                if not os.path.exists(lstm_labels_path):
                    raise ValueError('LSTM class labels are missing.')
                with open(lstm_labels_path, 'rb') as f:
                    labels = pickle.load(f)
                    self.lstm_labels = (
                        labels.classes_.tolist() if hasattr(labels, 'classes_') else list(labels)
                    )

                config_path = os.path.join('models', 'trained', 'lstm_model_config.json')
                if os.path.exists(config_path):
                    with open(config_path) as f:
                        cfg = json.load(f)
                        self.sequence_length = int(cfg.get('sequence_length', 30))
                        num_features = int(cfg.get('num_features', 63))
                else:
                    num_features = 63

                if tuple(model.input_shape[1:]) != (self.sequence_length, num_features) or num_features != 63:
                    raise ValueError(
                        f'Expected LSTM input ({self.sequence_length}, 63), '
                        f'got {tuple(model.input_shape[1:])}.'
                    )
                if not self.lstm_labels or int(model.output_shape[-1]) != len(self.lstm_labels):
                    raise ValueError(
                        'LSTM class label count does not match model output count.'
                    )
                if self.scaler is None or getattr(self.scaler, 'n_features_in_', None) != 63:
                    raise ValueError(
                        'The 63-feature CNN scaler required by LSTM inference is unavailable.'
                    )
                self.landmark_buffer = deque(maxlen=self.sequence_length)
                self.lstm_model = model
                print(f"✅ LSTM model loaded ({len(self.lstm_labels)} dynamic gestures)")
            except Exception as e:
                self.lstm_load_error = str(e)
                print(f"⚠️ Could not load LSTM model: {e}")
                self.lstm_model = None
        else:
            print("ℹ️ No LSTM model found — dynamic gestures disabled")
    
    def draw_landmarks(self, image: np.ndarray, landmarks_data: Dict = None) -> np.ndarray:
        """
        Draw hand landmarks on image
        
        Args:
            image: Input image
            landmarks_data: Pre-extracted landmarks (optional)
            
        Returns:
            Image with landmarks drawn
        """
        if landmarks_data is None:
            landmarks_data = self.extract_landmarks(image)
        
        if landmarks_data is None:
            return image
        
        annotated_image = image.copy()
        h, w, _ = annotated_image.shape
        
        # Draw landmarks from stored data (no re-processing)
        for hand_idx, landmarks in enumerate(landmarks_data['landmarks']):
            # Reshape to (21, 3) for drawing
            points = np.array(landmarks).reshape(-1, 3)
            pixel_points = []
            
            for point in points:
                px = int(point[0] * w)
                py = int(point[1] * h)
                pixel_points.append((px, py))
                # Draw landmark points
                cv2.circle(annotated_image, (px, py), 4, (0, 255, 0), -1)
                cv2.circle(annotated_image, (px, py), 6, (0, 0, 255), 1)
            
            # Draw hand connections (MediaPipe hand topology)
            connections = [
                (0,1),(1,2),(2,3),(3,4),       # Thumb
                (0,5),(5,6),(6,7),(7,8),       # Index
                (0,9),(9,10),(10,11),(11,12),  # Middle
                (0,13),(13,14),(14,15),(15,16),# Ring
                (0,17),(17,18),(18,19),(19,20),# Pinky
                (5,9),(9,13),(13,17)           # Palm
            ]
            
            for start_idx, end_idx in connections:
                if start_idx < len(pixel_points) and end_idx < len(pixel_points):
                    cv2.line(annotated_image, pixel_points[start_idx], pixel_points[end_idx], (0, 255, 0), 2)
            
            # Draw bounding box
            if 'bbox' in landmarks_data and hand_idx < len(landmarks_data['bbox']):
                bbox = landmarks_data['bbox'][hand_idx]
                x1, y1, x2, y2 = bbox
                x1, x2 = int(x1 * w), int(x2 * w)
                y1, y2 = int(y1 * h), int(y2 * h)
                
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw handedness label
                if 'handedness' in landmarks_data and hand_idx < len(landmarks_data['handedness']):
                    handedness = landmarks_data['handedness'][hand_idx]
                    cv2.putText(annotated_image, handedness, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_image
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Model information dictionary
        """
        if self.model is None:
            return {'error': 'No model loaded'}
        
        return {
            'model_type': 'CNN' if 'CNN' in str(type(self.model)) else 'Sequential',
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'num_parameters': self.model.count_params(),
            'num_classes': len(self.class_labels) if self.class_labels else 0,
            'class_labels': self.class_labels,
            'avg_prediction_time': np.mean(self.prediction_times) if self.prediction_times else 0,
            'model_loaded': self.cnn_model_loaded,
            'load_error': self.cnn_load_error
        }
    
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics
        
        Returns:
            Performance metrics dictionary
        """
        return {
            'avg_prediction_time': np.mean(self.prediction_times) if self.prediction_times else 0,
            'min_prediction_time': np.min(self.prediction_times) if self.prediction_times else 0,
            'max_prediction_time': np.max(self.prediction_times) if self.prediction_times else 0,
            'total_predictions': len(self.prediction_times),
            'fps_estimate': 1.0 / np.mean(self.prediction_times) if self.prediction_times else 0
        }
    
    def save_model(self, path: str):
        """
        Save the trained model
        
        Args:
            path: Path to save the model
        """
        if self.model:
            self.model.save(path)
            
            # Save class labels
            if self.class_labels:
                label_path = path.replace('.h5', '_labels.pkl')
                with open(label_path, 'wb') as f:
                    pickle.dump(self.class_labels, f)
            
            # Save scaler
            if self.scaler:
                scaler_path = path.replace('.h5', '_scaler.pkl')
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
            
            print(f"✅ Model saved to {path}")
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'hands'):
            self.hands.close()
        if self.bigru_holistic is not None:
            self.bigru_holistic.close()
        print("🧹 Gesture recognizer cleaned up")