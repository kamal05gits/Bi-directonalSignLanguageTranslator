import os
import sys
import json
import cv2
import numpy as np
import tensorflow as tf
import torch

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

VIDEO_PATH = (
    r"C:\Users\ELCOT\Downloads\CISLR\videos"
    r"\CISLR_v1.5-a_videos\0NMbSbEZ7G8_1.mp4"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "FINAL_DEMO_MODEL.keras"
)

LABELS_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "FINAL_DEMO_LABELS.json"
)

NORMALIZATION_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "FINAL_DEMO_NORMALIZATION.npz"
)

I3D_CODE_DIR = os.path.join(
    PROJECT_ROOT,
    "external",
    "WLASL",
    "code",
    "I3D"
)

I3D_WEIGHTS = (
    r"C:\Users\ELCOT\Downloads\Bi-directional Sign language translator"
    r"\external\WLASL\code\I3D\weights\archived\asl2000"
    r"\FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt"
)

# ============================================================
# CONFIGURATION
# ============================================================

NUM_INPUT_FRAMES = 90
TARGET_FEATURE_FRAMES = 32
IMAGE_SIZE = 224

print("=" * 70)
print("       CISLR REAL-TIME INDIAN SIGN LANGUAGE DEMO")
print("=" * 70)

# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    VIDEO_PATH,
    MODEL_PATH,
    LABELS_PATH,
    NORMALIZATION_PATH,
    os.path.join(I3D_CODE_DIR, "pytorch_i3d.py"),
    I3D_WEIGHTS,
]

print("\nChecking required files...")

for path in required_files:
    if not os.path.exists(path):
        print("\nERROR: File not found:")
        print(path)
        sys.exit(1)

print("All required files found.")

# ============================================================
# LOAD LABELS
# ============================================================

print("\n[1/6] Loading labels...")

with open(
    LABELS_PATH,
    "r",
    encoding="utf-8"
) as f:
    labels_data = json.load(f)

if isinstance(labels_data, list):
    labels = labels_data

elif isinstance(labels_data, dict):

    if "labels" in labels_data:
        labels = labels_data["labels"]

    elif all(str(k).isdigit() for k in labels_data.keys()):
        labels = [
            labels_data[str(i)]
            for i in range(len(labels_data))
        ]

    else:
        labels = list(labels_data.values())

else:
    raise RuntimeError(
        "Unsupported labels JSON format."
    )

labels = list(labels)

print("Number of labels:", len(labels))

# ============================================================
# LOAD NORMALIZATION
# ============================================================

print("\n[2/6] Loading normalization...")

norm = np.load(NORMALIZATION_PATH)

print("Arrays:", norm.files)

if "mean" not in norm or "std" not in norm:
    raise RuntimeError(
        "Normalization file must contain 'mean' and 'std'."
    )

mean = norm["mean"].astype(np.float32)
std = norm["std"].astype(np.float32)

std = np.where(
    std < 1e-6,
    1.0,
    std
)

print("Mean shape:", mean.shape)
print("Std shape :", std.shape)

# ============================================================
# LOAD CLASSIFIER
# ============================================================

print("\n[3/6] Loading classifier...")

classifier = tf.keras.models.load_model(
    MODEL_PATH
)

print("Classifier loaded.")
print("Input shape :", classifier.input_shape)
print("Output shape:", classifier.output_shape)

# ============================================================
# LOAD I3D
# ============================================================

print("\n[4/6] Loading I3D...")

sys.path.insert(
    0,
    I3D_CODE_DIR
)

from pytorch_i3d import InceptionI3d

device = torch.device("cpu")

i3d = InceptionI3d(
    num_classes=2000,
    in_channels=3
)

checkpoint = torch.load(
    I3D_WEIGHTS,
    map_location="cpu"
)

if isinstance(checkpoint, dict):

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint

clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        key = key[7:]

    clean_state_dict[key] = value

missing, unexpected = i3d.load_state_dict(
    clean_state_dict,
    strict=False
)

print("I3D checkpoint loaded.")
print("Missing keys   :", len(missing))
print("Unexpected keys:", len(unexpected))

i3d.to(device)
i3d.eval()

# ============================================================
# READ VIDEO
# ============================================================

print("\n[5/6] Reading CISLR video...")

print("Video:")
print(VIDEO_PATH)

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open CISLR video."
    )

frames = []

while True:

    ok, frame = cap.read()

    if not ok:
        break

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    frames.append(frame)

cap.release()

print("Decoded frames:", len(frames))

if len(frames) == 0:
    raise RuntimeError(
        "Video contains no frames."
    )

# ============================================================
# SAMPLE FRAMES
# ============================================================

sample_indices = np.linspace(
    0,
    len(frames) - 1,
    NUM_INPUT_FRAMES
).astype(np.int32)

sampled_frames = [
    frames[i]
    for i in sample_indices
]

print(
    "Sampled frames:",
    len(sampled_frames)
)

# ============================================================
# PREPROCESS
# ============================================================

processed = []

for frame in sampled_frames:

    h, w = frame.shape[:2]

    scale = IMAGE_SIZE / min(
        h,
        w
    )

    new_w = int(
        round(w * scale)
    )

    new_h = int(
        round(h * scale)
    )

    resized = cv2.resize(
        frame,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    y0 = max(
        (new_h - IMAGE_SIZE) // 2,
        0
    )

    x0 = max(
        (new_w - IMAGE_SIZE) // 2,
        0
    )

    cropped = resized[
        y0:y0 + IMAGE_SIZE,
        x0:x0 + IMAGE_SIZE
    ]

    cropped = cv2.resize(
        cropped,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    cropped = cropped.astype(
        np.float32
    )

    cropped = (
        cropped / 127.5
    ) - 1.0

    processed.append(cropped)

video_np = np.stack(
    processed
)

print(
    "Preprocessed video:",
    video_np.shape
)

# ============================================================
# I3D INPUT
# ============================================================

# T,H,W,C
# -> C,T,H,W
# -> B,C,T,H,W

i3d_input = np.transpose(
    video_np,
    (3, 0, 1, 2)
)

i3d_input = torch.from_numpy(
    i3d_input
).unsqueeze(0).float()

print(
    "I3D input:",
    tuple(i3d_input.shape)
)

# ============================================================
# FEATURE EXTRACTION
# ============================================================

print(
    "\nExtracting I3D features..."
)

with torch.no_grad():

    output = i3d.extract_features(
        i3d_input
    )

print(
    "Raw I3D output:",
    tuple(output.shape)
)

# B,C,T,H,W -> B,C,T
if output.ndim == 5:

    output = output.mean(
        dim=(-1, -2)
    )

    features = (
        output[0]
        .cpu()
        .numpy()
        .T
    )

elif output.ndim == 3:

    features = (
        output[0]
        .cpu()
        .numpy()
        .T
    )

else:

    raise RuntimeError(
        "Unexpected I3D output shape: "
        + str(output.shape)
    )

print(
    "I3D feature sequence:",
    features.shape
)

# ============================================================
# CHECK FEATURE DIMENSION
# ============================================================

if features.shape[1] != 1024:

    raise RuntimeError(
        "Expected 1024 feature dimensions, "
        f"but received {features.shape[1]}."
    )

# ============================================================
# TEMPORAL RESAMPLING
# ============================================================

source_length = features.shape[0]

source_positions = np.linspace(
    0,
    source_length - 1,
    source_length
)

target_positions = np.linspace(
    0,
    source_length - 1,
    TARGET_FEATURE_FRAMES
)

resampled = np.zeros(
    (
        TARGET_FEATURE_FRAMES,
        1024
    ),
    dtype=np.float32
)

for d in range(1024):

    resampled[:, d] = np.interp(
        target_positions,
        source_positions,
        features[:, d]
    )

features = resampled

print(
    "Final feature sequence:",
    features.shape
)

if features.shape != (
    32,
    1024
):

    raise RuntimeError(
        "Classifier requires (32,1024), "
        f"but received {features.shape}."
    )

# ============================================================
# NORMALIZATION
# ============================================================

features = (
    features - mean.reshape(1, 1024)
) / std.reshape(1, 1024)

features = np.nan_to_num(
    features,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

X = features[
    np.newaxis,
    ...
]

print(
    "Classifier input:",
    X.shape
)

# ============================================================
# PREDICTION
# ============================================================

print(
    "\n[6/6] Running prediction..."
)

probabilities = classifier.predict(
    X,
    verbose=0
)[0]

top_indices = np.argsort(
    probabilities
)[::-1][:5]

# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n")
print("=" * 70)
print("                 PREDICTION RESULT")
print("=" * 70)

top1 = int(
    top_indices[0]
)

print("\nTop-1 Prediction:")
print(
    "   ",
    labels[top1]
)

print("\nConfidence:")
print(
    f"    {probabilities[top1] * 100:.2f}%"
)

print("\nTop-5 Predictions:")

for rank, idx in enumerate(
    top_indices,
    start=1
):

    idx = int(idx)

    print(
        f"    {rank}. "
        f"{labels[idx]} "
        f"({probabilities[idx] * 100:.2f}%)"
    )

print("\n")
print("=" * 70)
print("                 DEMO COMPLETE")
print("=" * 70)