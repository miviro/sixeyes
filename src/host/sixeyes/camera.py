from __future__ import annotations

import cv2
import numpy as np

from .config import CAMERA_BACKEND, CAMERA_INDEX, DEADBAND_PIXELS
from .models import TrackerTelemetry
from .tracker import PanTiltTracker


def open_camera() -> cv2.VideoCapture:
    capture = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    if capture.isOpened():
        return capture

    capture.release()
    capture = cv2.VideoCapture(CAMERA_INDEX)
    if capture.isOpened():
        return capture

    # Fix: release before raising so the VideoCapture object is not leaked
    capture.release()
    raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")


def annotate_frame(
    frame: np.ndarray,
    telemetry: TrackerTelemetry,
    tracker: PanTiltTracker,
    fps: float,
) -> np.ndarray:
    overlay = frame.copy()
    center = (int(round(tracker.center_x)), int(round(tracker.center_y)))

    cv2.drawMarker(
        overlay,
        center,
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=1,
    )
    cv2.circle(overlay, center, DEADBAND_PIXELS, (255, 255, 255), 1)

    for detection in telemetry.detections:
        selected = detection == telemetry.target
        color = (0, 220, 0) if selected else (0, 165, 255)
        cv2.rectangle(
            overlay,
            (int(round(detection.x1)), int(round(detection.y1))),
            (int(round(detection.x2)), int(round(detection.y2))),
            color,
            2,
        )
        label = f"{detection.label} {detection.confidence:.2f}"
        cv2.putText(
            overlay,
            label,
            (int(round(detection.x1)), max(18, int(round(detection.y1)) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    if telemetry.estimate is not None:
        cv2.circle(
            overlay,
            (int(round(telemetry.estimate[0])), int(round(telemetry.estimate[1]))),
            5,
            (0, 255, 255),
            1,
        )

    if telemetry.aim_point is not None:
        cv2.circle(
            overlay,
            (int(round(telemetry.aim_point[0])), int(round(telemetry.aim_point[1]))),
            6,
            (255, 200, 0),
            2,
        )
        cv2.line(
            overlay,
            center,
            (int(round(telemetry.aim_point[0])), int(round(telemetry.aim_point[1]))),
            (255, 200, 0),
            1,
        )

    status_color = (0, 220, 0) if telemetry.state == "TRACK" else (0, 165, 255)
    text_lines = [
        f"State: {telemetry.state}",
        f"Yaw: {telemetry.current_yaw:5.1f}  Pitch: {telemetry.current_pitch:5.1f}",
        f"Error(px): {telemetry.error_x:+6.1f}  {telemetry.error_y:+6.1f}",
        f"Lost: {telemetry.lost_frames:02d}  FPS: {fps:4.1f}",
    ]

    for index, text in enumerate(text_lines):
        cv2.putText(
            overlay,
            text,
            (12, 28 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2,
            cv2.LINE_AA,
        )

    return overlay
