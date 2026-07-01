import cv2
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque

from slowfast.config.defaults import get_cfg
from slowfast.models import build_model
from slowfast.utils.checkpoint import load_checkpoint


# 1. GLOBAL CONFIG (MATCH TRAINING)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# X3D input requirements
NUM_FRAMES = 13
SAMPLING_RATE = 6
CROP_SIZE = 182

# Detection threshold (from evaluation)
THRESHOLD = 0.05

CFG_PATH = ""
CHECKPOINT_PATH = ""

# Normalization (Kinetics / X3D)
MEAN = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
STD  = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1)

WINDOW = 5
conf_buffer = deque(maxlen=WINDOW)

# Class index → label (same order as training)
CLASS_NAMES = [
    "call for help", "chase someone", "chest discomfort",
    "cross arms", "drag someone", "escape",
    "hit someone", "hostage", "kicking",
    "lying", "point", "punch",
    "punching", "pushing", "robbery",
    "run", "sitting", "slap",
    "stab", "stagger", "stalk",
    "steal", "knife threat", "gun threat",
    "vomit", "wave", "wear a mask"
]


# 2. LOAD PySlowFast MODEL 
def load_pyslowfast_model(cfg_path, checkpoint_path):
    """
    Load X3D model exactly as trained using PySlowFast.
    """

    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)

    # Inference-only overrides
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True
    cfg.TEST.BATCH_SIZE = 1
    cfg.NUM_GPUS = 1 if torch.cuda.is_available() else 0

    cfg.freeze()

    model = build_model(cfg)
    model = model.to(DEVICE)
    model.eval()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" in checkpoint:
        checkpoint = checkpoint["model_state"]

    missing, unexpected = model.load_state_dict(checkpoint, strict=False)

    print("[MODEL LOAD]")
    print("  Missing keys:", len(missing))
    print("  Unexpected keys:", len(unexpected))

    assert len(missing) == 0 and len(unexpected) == 0, \
        "Model and checkpoint are NOT aligned"

    print("[OK] PySlowFast X3D model loaded correctly\n")
    return model


# 3. VIDEO LOADING
def load_video(video_path):
    """
    Load video → Tensor [T, C, H, W]
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames read from {video_path}")

    frames = np.stack(frames)                 # [T, H, W, C]
    frames = torch.from_numpy(frames)         # uint8
    frames = frames.permute(0, 3, 1, 2)       # [T, C, H, W]

    return frames


# 4. PREPROCESSING (CRITICAL)
def preprocess_video(frames):
    """
    frames: Tensor [T, C, H, W]
    returns: Tensor [C, T, H, W]
    """

    required = NUM_FRAMES * SAMPLING_RATE
    if frames.shape[0] < required:
        repeat = (required // frames.shape[0]) + 1
        frames = frames.repeat(repeat, 1, 1, 1)

    # Temporal sampling
    frames = frames[::SAMPLING_RATE][:NUM_FRAMES]

    # Normalize to [0,1]
    frames = frames.float() / 255.0

    # Resize short side
    T, C, H, W = frames.shape
    scale = CROP_SIZE / min(H, W)
    new_h, new_w = int(H * scale), int(W * scale)

    frames = F.interpolate(
        frames,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False
    )

    # Center crop
    top = (new_h - CROP_SIZE) // 2
    left = (new_w - CROP_SIZE) // 2
    frames = frames[:, :, top:top+CROP_SIZE, left:left+CROP_SIZE]

    # Mean / std
    frames = (frames - MEAN) / STD

    # [T,C,H,W] → [C,T,H,W]
    frames = frames.permute(1, 0, 2, 3)

    return frames

def temporal_decision(conf):
    conf_buffer.append(conf)
    avg_conf = sum(conf_buffer) / len(conf_buffer)
    return avg_conf


# 5. UAV ACTION RECOGNITION (FINAL LOGIC)
def run_action_recognition(clip_path, model, threshold=THRESHOLD):
    """
    Binary suspicious-action detection for UAV surveillance.

    Returns:
        action_label (str)
        suspicious (bool)
        confidence (float)
    """

    frames = load_video(clip_path)
    clip = preprocess_video(frames)          # [C, T, H, W]
    clip = clip.unsqueeze(0)              # [1, C, T, H, W]
    inputs = [clip]                       # <-- CRITICAL

    with torch.no_grad():
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)[0]

    max_conf, idx = torch.max(probs, dim=0)

    confidence = float(max_conf.item())
    #suspicious = confidence >= threshold

    avg_conf = temporal_decision(confidence)
    suspicious = avg_conf >= THRESHOLD

    action_label = CLASS_NAMES[idx.item()] if suspicious else "none"

    return action_label, suspicious, confidence


def sliding_window_action_recognition(
    video_path,
    model,
    stride_frames=6,
    threshold=THRESHOLD
):
    """
    Sliding-window + temporal aggregation for full video.
    Returns:
        action_label (str)
        suspicious (bool)
        confidence (float)
    """

    frames = load_video(video_path)
    T = frames.shape[0]
    window_size = NUM_FRAMES * SAMPLING_RATE

    best_conf = 0.0
    best_idx = None

    for start in range(0, T - window_size + 1, stride_frames):
        clip_frames = frames[start:start + window_size]
        clip = preprocess_video(clip_frames)
        clip = clip.unsqueeze(0)
        inputs = [clip]

        with torch.no_grad():
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)[0]

        conf, idx = torch.max(probs, dim=0)
        conf = conf.item()

        # best_conf = max(best_conf, conf)
        # if conf == best_conf:
        #     best_idx = idx.item()

        if conf > best_conf:
            best_conf = conf
            best_idx = idx.item()

    suspicious = best_conf >= threshold
    label = CLASS_NAMES[best_idx] if suspicious else "none"

    return label, suspicious, best_conf

