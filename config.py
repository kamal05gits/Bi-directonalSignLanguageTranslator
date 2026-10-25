"""
Configuration file for ISL Gesture Recognition System
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'isl-gesture-recognition-secret-key'
    DEBUG = False
    TESTING = False
    
    # Model Configuration
    MODEL_PATH = os.path.join('models', 'trained')
    CNN_MODEL_PATH = os.path.join(MODEL_PATH, 'cnn_model.h5')
    
    # Gesture Recognition Settings
    CONFIDENCE_THRESHOLD = 0.6
    EMERGENCY_CONFIDENCE_THRESHOLD = 0.65
    PREDICTION_SMOOTHING_WINDOW = 5
    PREDICTION_MIN_STABLE_FRAMES = 3
    PREDICTION_COMMIT_COOLDOWN = 2.0
    SAME_GESTURE_COOLDOWN = 4.0
    HAND_DETECTION_CONFIDENCE = 0.5  # Lowered for better detection
    HAND_TRACKING_CONFIDENCE = 0.3   # Lowered for better tracking
    MAX_NUM_HANDS = 2
    
    # Data Paths
    DATA_DIR = 'data'
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    DATASETS_DIR = os.path.join(DATA_DIR, 'datasets')
    
    # Static Files
    STATIC_DIR = 'static'
    UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'}
    
    # Text-to-Speech Configuration
    TTS_ENGINE = 'pyttsx3'  # or 'gtts'
    TTS_RATE = 150
    TTS_VOLUME = 0.8
    TTS_VOICE_INDEX = 0  # 0 for male, 1 for female
    
    # NLP Configuration
    NLTK_DATA_PATH = os.path.join(os.getcwd(), 'nltk_data')
    SPACY_MODEL = 'en_core_web_sm'
    
    # Performance Monitoring
    LOG_PERFORMANCE = True
    LOG_FILE = 'logs/performance.log'
    FPS_LOG_INTERVAL = 30  # Log FPS every 30 seconds
    
    # Gesture Classes (81 target classes — alphabetical order)
    # NOTE: The trained CNN model currently recognizes 61 of these 81 classes.
    # Missing from trained model: BROTHER, FAMILY, FATHER, FOOD, FRIEND, GOODBYE,
    # HAPPY, HOUSE, MONEY, MORNING, MOTHER, NIGHT, PLEASE, SAD, SISTER, SORRY,
    # TEACHER, THANKYOU, TIME, WATER.
    # The model loads its actual labels from cnn_model_labels.pkl independently.
    # To add support for all 81, collect data and retrain via /collect page.
    GESTURE_CLASSES = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "A", "AFRAID", "AGREE", "ASSISTANCE", "B", "BAD", "BECOME",
        "BROTHER", "C", "COLLEGE", "D", "DOCTOR", "E", "F", "FAMILY",
        "FATHER", "FOOD", "FRIEND", "FROM", "G", "GOODBYE", "H",
        "HAPPY", "HELLO", "HOUSE", "I", "J", "K", "L", "LOVE", "M",
        "MONEY", "MORNING", "MOTHER", "N", "NIGHT", "NO", "O", "P",
        "PAIN", "PLEASE", "PRAY", "Q", "R", "S", "SAD", "SECONDARY",
        "SISTER", "SKIN", "SMALL", "SORRY", "SPACE", "SPECIFIC",
        "STAND", "T", "TEACHER", "THANKYOU", "TIME", "TODAY", "U", "V",
        "W", "WARN", "WATER", "WHICH", "WORK", "X", "Y", "YES",
        "YOU", "Z"
    ]
    
    NUM_CLASSES = len(GESTURE_CLASSES)  # 81 target classes

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_PERFORMANCE = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    CONFIDENCE_THRESHOLD = 0.8
    LOG_PERFORMANCE = True

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    CONFIDENCE_THRESHOLD = 0.3  # Lower threshold for testing
    DEBUG = True

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}