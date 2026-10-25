"""
LSTM Model for Dynamic Gesture Recognition
Trains on sequences of MediaPipe hand landmarks to recognize moving gestures.
Each sample is a sequence of 30 frames × 63 features (21 landmarks × 3 coords).
"""
import numpy as np
import sys
import os

# Fix Windows console encoding for emoji characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import json
import pickle
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Config
SEQUENCE_LENGTH = 30  # frames per gesture
NUM_FEATURES = 63     # 21 landmarks × 3 (x, y, z)
DATA_DIR = os.path.join('data', 'sequences')
MODEL_DIR = os.path.join('models', 'trained')
CHECKPOINT_DIR = os.path.join('models', 'checkpoints')


class LSTMTrainer:
    """Train LSTM model for dynamic gesture recognition"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.classes = []
        
    def load_data(self):
        """Load sequence data from data/sequences/<gesture_name>/*.npy"""
        X, y = [], []
        
        if not os.path.exists(DATA_DIR):
            print(f"❌ No sequence data found at {DATA_DIR}")
            print("   Use the Collect page to record dynamic gesture sequences first.")
            return None, None
        
        gestures = sorted([d for d in os.listdir(DATA_DIR) 
                          if os.path.isdir(os.path.join(DATA_DIR, d))])
        
        if len(gestures) < 2:
            print(f"❌ Need at least 2 gesture classes, found {len(gestures)}")
            return None, None
        
        print(f"📂 Found {len(gestures)} dynamic gesture classes")
        
        for gesture in gestures:
            gesture_dir = os.path.join(DATA_DIR, gesture)
            sequences = [f for f in os.listdir(gesture_dir) if f.endswith('.npy')]
            
            for seq_file in sequences:
                seq = np.load(os.path.join(gesture_dir, seq_file))
                if seq.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
                    X.append(seq)
                    y.append(gesture)
                    
            print(f"  {gesture}: {len(sequences)} sequences")
        
        if len(X) == 0:
            print("❌ No valid sequences found")
            return None, None
            
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        if not np.all(np.isfinite(X)):
            raise ValueError('Sequence data contains non-finite landmark features.')

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes = self.label_encoder.classes_.tolist()
        class_counts = np.bincount(y_encoded)
        test_size = int(np.ceil(len(X) * 0.2))
        if (
            len(self.classes) < 2
            or np.any(class_counts < 2)
            or test_size < len(self.classes)
            or len(X) - test_size < len(self.classes)
        ):
            print(
                '❌ Need at least two classes with enough samples for a stratified '
                'train/test split.'
            )
            return None, None
        y_onehot = keras.utils.to_categorical(y_encoded)
        
        print(f"✅ Loaded {len(X)} sequences, {len(self.classes)} classes")
        print(f"   Shape: {X.shape}")
        return X, y_onehot
    
    def build_model(self, num_classes):
        """Build bidirectional LSTM model"""
        self.model = keras.Sequential([
            Bidirectional(LSTM(128, return_sequences=True), 
                         input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
            BatchNormalization(),
            Dropout(0.3),
            
            Bidirectional(LSTM(64)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model.summary()
        return self.model
    
    def train(self, epochs=100, batch_size=32):
        """Full training pipeline"""
        print("🚀 LSTM Dynamic Gesture Training")
        print("=" * 50)
        
        # Load data
        X, y = self.load_data()
        if X is None:
            return None
        
        num_classes = y.shape[1]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
        )
        
        print(f"📈 Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Build
        self.build_model(num_classes)
        
        # Callbacks
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        callbacks = [
            ModelCheckpoint(
                os.path.join(CHECKPOINT_DIR, 'best_lstm.h5'),
                monitor='val_accuracy', save_best_only=True, verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy', patience=15, 
                restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, verbose=1
            )
        ]
        
        # Train
        start = datetime.now()
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        duration = datetime.now() - start
        
        # Evaluate
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\n✅ Training completed in {duration}")
        print(f"📊 Test accuracy: {accuracy:.4f}")
        
        # Save
        self.save_model()
        
        return history
    
    def save_model(self):
        """Save model, labels, and config"""
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        model_path = os.path.join(MODEL_DIR, 'lstm_model.h5')
        self.model.save(model_path)
        
        labels_path = os.path.join(MODEL_DIR, 'lstm_model_labels.pkl')
        with open(labels_path, 'wb') as f:
            pickle.dump(self.classes, f)
        
        config_path = os.path.join(MODEL_DIR, 'lstm_model_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'num_classes': len(self.classes),
                'sequence_length': SEQUENCE_LENGTH,
                'num_features': NUM_FEATURES,
                'classes': self.classes
            }, f, indent=2)
        
        print(f"💾 LSTM model saved to {model_path}")
        print(f"   Classes: {self.classes}")


def main():
    trainer = LSTMTrainer()
    trainer.train()


if __name__ == '__main__':
    main()
