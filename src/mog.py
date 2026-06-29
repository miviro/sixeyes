import cv2
import numpy as np
from config import MOG_DELTA_FRAMES, MOG_VAR_THRESHOLD, MOG_DETECT_SHADOWS, MOG_SCALE_W, MOG_SCALE_H


def make_mog():
    return cv2.createBackgroundSubtractorMOG2(
        history=MOG_DELTA_FRAMES,
        varThreshold=MOG_VAR_THRESHOLD,
        detectShadows=MOG_DETECT_SHADOWS,
    )


def apply_mog(mog, frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (MOG_SCALE_W, MOG_SCALE_H), interpolation=cv2.INTER_LINEAR)
    mask = mog.apply(small)
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
