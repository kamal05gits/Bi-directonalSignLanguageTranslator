#!/usr/bin/env python3
"""
Quick test to verify gesture recognizer initialization
"""
import importlib.util
import os
import sys
from pathlib import Path


def _prefer_local_venv():
    """Use the bundled virtual environment when the system interpreter cannot import project dependencies."""
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

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_initialization():
    """Test the initialization of components"""
    print("🔍 Testing ISL Gesture Recognition System Components")
    print("=" * 50)
    
    try:
        from config import config
        print("✅ Config imported successfully")
        
        # Test configuration
        app_config = config['development']()
        print("📱 Camera index: Not defined in DevelopmentConfig")
        print(f"🎯 Detection confidence: {app_config.HAND_DETECTION_CONFIDENCE}")
        print(f"📊 Model path: {app_config.CNN_MODEL_PATH}")
        print(f"🔢 Number of classes: {app_config.NUM_CLASSES}")
        
        # Test model file existence
        if os.path.exists(app_config.CNN_MODEL_PATH):
            print("✅ CNN model file exists")
        else:
            print("❌ CNN model file missing")
            return False
        
        # Test gesture recognizer
        from src.models.gesture_recognizer import GestureRecognizer
        print("✅ GestureRecognizer imported successfully")
        
        model_config = {
            'MAX_NUM_HANDS': app_config.MAX_NUM_HANDS,
            'HAND_DETECTION_CONFIDENCE': app_config.HAND_DETECTION_CONFIDENCE,
            'HAND_TRACKING_CONFIDENCE': app_config.HAND_TRACKING_CONFIDENCE
        }
        
        recognizer = GestureRecognizer(
            model_path=app_config.CNN_MODEL_PATH,
            config=model_config
        )
        print("✅ Gesture recognizer initialized")
        
        # Test model info
        model_info = recognizer.get_model_info()
        print(f"📊 Model loaded: {model_info.get('model_loaded', False)}")
        print(f"🎯 Number of classes: {model_info.get('num_classes', 'Unknown')}")
        
        # Test camera
        import cv2
        camera_index = getattr(app_config, "CAMERA_INDEX", 0)
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
           print(f"✅ Camera opened successfully at index {camera_index}")
        else:
          print(f"❌ Could not open camera at index {camera_index}")
        cap.release()
        if cap.isOpened():
            print("✅ Camera accessible")
            cap.release()
        else:
            print("❌ Camera not accessible")
        
        print("\n🎉 All components initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_initialization()
    sys.exit(0 if success else 1)