# SignOra - Indian Sign Language Recognition

## Model

BiGRU classifier using MediaPipe Holistic landmarks.

## Input

40 frames × 225 features per frame.

### Features per frame

- Pose: 33 × 3 = 99
- Left hand: 21 × 3 = 63
- Right hand: 21 × 3 = 63

Total = 225 features.

Before normalization, multiply each landmark's Y coordinate by
`frame_height / frame_width`, matching the training-time aspect-ratio correction.

## Required inference files

- best_isl_bigru.keras
- holistic_landmarker.task
- classes.json
- mean.npy
- std.npy

## Prediction pipeline

Video/Webcam
    ↓
MediaPipe Holistic
    ↓
Pose + Left Hand + Right Hand
    ↓
225 features/frame
    ↓
Aspect-ratio correction (Y × frame height / frame width)
    ↓
40-frame sequence
    ↓
Normalization
    ↓
BiGRU
    ↓
262-class prediction
    ↓
Indian Sign Language word

## Evaluation

See classification_report.txt.
