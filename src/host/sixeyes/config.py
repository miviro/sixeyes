from __future__ import annotations

import cv2

# Hardware
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
MODEL_PATH = "yolo11m.pt"
TARGET_LABEL = "bottle"
CAMERA_INDEX = 0
CAMERA_BACKEND = cv2.CAP_V4L2
FRAME_ROTATION = None

# Servo travel limits (degrees)
YAW_MIN, YAW_MAX = 0.0, 160.0
PITCH_MIN, PITCH_MAX = 10.0, 140.0

# Deadband around frame centre treated as locked
DEADBAND_PIXELS = 20

# Frames without detection before falling back to search
LOST_FRAMES_THRESHOLD = 20

# Search pattern: yaw triangle wave, pitch sinusoid around centre
SWEEP_SPEED = 0.01
SWEEP_PITCH_CENTER = (PITCH_MIN + PITCH_MAX) / 2.0
SWEEP_PITCH_AMP = 20.0
SWEEP_PITCH_FREQ = 6.0

# PID gains (servo angle delta in degrees per control update)
YAW_KP, YAW_KI, YAW_KD = 0.01, 0.0, 0.02
PITCH_KP, PITCH_KI, PITCH_KD = 0.01, 0.0, 0.02
INTEGRAL_CLAMP = 20.0

# Kalman filter noise
KF_PROCESS_NOISE = 5.0
KF_MEAS_NOISE = 5.0

# Pixels-per-degree calibration (landscape orientation)
CAMERA_FOV_X = 62.0
CAMERA_FOV_Y = 48.0

# Runtime tuning
CONTROL_REFERENCE_FPS = 30.0
MIN_FRAME_DT_SECONDS = 1.0 / 240.0
MAX_FRAME_DT_SECONDS = 0.25
PREDICTION_LEAD_SECONDS = 0.06
MAX_YAW_SPEED_DPS = 220.0
MAX_PITCH_SPEED_DPS = 220.0
DERIVATIVE_SMOOTHING = 0.6
TARGET_CONFIDENCE_THRESHOLD = 0.25
MIN_BOX_AREA_RATIO = 0.0008
ASSOCIATION_DISTANCE_RATIO = 0.35
WINDOW_NAME = "SixEyes Tracker"


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
