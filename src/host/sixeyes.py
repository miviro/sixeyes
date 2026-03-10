from __future__ import annotations

import time
from typing import Optional

import cv2

from sixeyes.camera import annotate_frame, open_camera
from sixeyes.config import (
    BAUD_RATE,
    FRAME_ROTATION,
    MODEL_PATH,
    PITCH_MAX,
    PITCH_MIN,
    SERIAL_PORT,
    WINDOW_NAME,
    YAW_MAX,
    YAW_MIN,
    clamp,
)
from sixeyes.detector import YoloDetector
from sixeyes.serial_link import ESP32SerialLink
from sixeyes.tracker import PanTiltTracker


def main() -> None:
    detector = YoloDetector(MODEL_PATH)
    serial_link: Optional[ESP32SerialLink] = None
    capture: Optional[cv2.VideoCapture] = None
    fps_ema = 0.0

    try:
        capture = open_camera()
        serial_link = ESP32SerialLink(SERIAL_PORT, BAUD_RATE)

        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        if FRAME_ROTATION is not None:
            frame = cv2.rotate(frame, FRAME_ROTATION)
        tracker = PanTiltTracker(frame.shape[1], frame.shape[0])
        # Reset timer after all setup (including the 2 s serial sleep) so the
        # first loop iteration gets a realistic dt instead of a multi-second spike.
        previous_time = time.monotonic()

        while True:
            ok, frame = capture.read()
            now = time.monotonic()
            dt = now - previous_time
            previous_time = now

            if not ok:
                raise RuntimeError("Camera read failed")

            if FRAME_ROTATION is not None:
                frame = cv2.rotate(frame, FRAME_ROTATION)

            detections = detector.detect(frame)
            telemetry = tracker.update(detections, dt)
            serial_link.send(telemetry.current_pitch, telemetry.current_yaw)

            instantaneous_fps = 1.0 / max(dt, 1e-6)
            fps_ema = instantaneous_fps if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * instantaneous_fps)

            annotated = annotate_frame(frame, telemetry, tracker, fps_ema)
            cv2.imshow(WINDOW_NAME, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        neutral_pitch = clamp(90.0, PITCH_MIN, PITCH_MAX)
        neutral_yaw = clamp(90.0, YAW_MIN, YAW_MAX)
        try:
            if serial_link is not None:
                serial_link.send(neutral_pitch, neutral_yaw)
        except Exception:
            pass
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()
        if serial_link is not None:
            serial_link.close()


if __name__ == "__main__":
    main()
