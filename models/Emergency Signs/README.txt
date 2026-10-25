SignOra Emergency Sign Recognition

Files:
- isl_emergency_lstm.keras: trained 8-class LSTM
- class_labels.json: model output labels
- hand_landmarker.task: MediaPipe hand detection model
- live_signora.py: webcam inference script
- requirements.txt: Python dependencies

Run on a local computer with a webcam:
1. Install Python 3.11 if possible.
2. Create a virtual environment.
3. Install dependencies from requirements.txt.
4. Run: python live_signora.py

The model recognizes only the eight emergency signs in this dataset.
It is an experimental classifier, not a validated emergency safety system.
Feed frames to the model in their original, unmirrored camera orientation.
Mirroring is for display only.
