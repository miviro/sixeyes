from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np
import serial
from ultralytics import YOLO

# --- Configuration carried over from the previous script ---
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
MODEL_PATH = "faces.pt"
CAMERA_INDEX = 0
CAMERA_BACKEND = cv2.CAP_V4L2
FRAME_ROTATION = cv2.ROTATE_90_COUNTERCLOCKWISE

# Servo travel limits (degrees)
YAW_MIN, YAW_MAX = 0.0, 160.0
PITCH_MIN, PITCH_MAX = 10.0, 140.0

# Deadband around frame centre treated as locked
DEADBAND_PIXELS = 20

# Frames without detection before falling back to search
LOST_FRAMES_THRESHOLD = 20

# Search pattern: pitch triangle wave, yaw sinusoid around centre
SWEEP_SPEED = 0.05
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

# Pixels-per-degree calibration after 90 degree CCW rotation
CAMERA_FOV_X = 62.0
CAMERA_FOV_Y = 48.0

# --- Additional runtime tuning ---
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


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: Optional[int]
    label: str

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) * 0.5

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) * 0.5

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class TrackerTelemetry:
    state: str
    detections: Sequence[Detection]
    target: Optional[Detection]
    estimate: Optional[tuple[float, float]]
    aim_point: Optional[tuple[float, float]]
    error_x: float
    error_y: float
    current_pitch: float
    current_yaw: float
    lost_frames: int


class ConstantVelocityKalmanFilter:
    def __init__(self, process_noise: float, measurement_noise: float):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state = np.zeros((4, 1), dtype=np.float64)
        self.covariance = np.eye(4, dtype=np.float64)
        self.initialized = False

    def reset(self) -> None:
        self.state = np.zeros((4, 1), dtype=np.float64)
        self.covariance = np.eye(4, dtype=np.float64)
        self.initialized = False

    def initialize(self, x: float, y: float) -> None:
        self.state[:, 0] = [x, y, 0.0, 0.0]
        self.covariance = np.diag([
            self.measurement_noise,
            self.measurement_noise,
            100.0,
            100.0,
        ]).astype(np.float64)
        self.initialized = True

    def _transition_matrix(self, dt: float) -> np.ndarray:
        return np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def _process_covariance(self, dt: float) -> np.ndarray:
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = self.process_noise
        return q * np.array([
            [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
            [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
            [dt3 / 2.0, 0.0, dt2, 0.0],
            [0.0, dt3 / 2.0, 0.0, dt2],
        ], dtype=np.float64)

    def predict(self, dt: float) -> Optional[tuple[float, float]]:
        if not self.initialized:
            return None

        transition = self._transition_matrix(dt)
        self.state = transition @ self.state
        self.covariance = (
            transition @ self.covariance @ transition.T
            + self._process_covariance(dt)
        )
        return self.position()

    def correct(self, x: float, y: float) -> tuple[float, float]:
        if not self.initialized:
            self.initialize(x, y)
            return self.position()

        measurement = np.array([[x], [y]], dtype=np.float64)
        measurement_matrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)
        measurement_covariance = np.eye(2, dtype=np.float64) * self.measurement_noise

        innovation = measurement - measurement_matrix @ self.state
        innovation_covariance = (
            measurement_matrix @ self.covariance @ measurement_matrix.T
            + measurement_covariance
        )
        kalman_gain = (
            self.covariance
            @ measurement_matrix.T
            @ np.linalg.inv(innovation_covariance)
        )

        identity = np.eye(4, dtype=np.float64)
        self.state = self.state + kalman_gain @ innovation
        correction = identity - kalman_gain @ measurement_matrix
        self.covariance = (
            correction @ self.covariance @ correction.T
            + kalman_gain @ measurement_covariance @ kalman_gain.T
        )
        return self.position()

    def update(
        self,
        dt: float,
        measurement: Optional[tuple[float, float]],
    ) -> Optional[tuple[float, float]]:
        dt = clamp(dt, MIN_FRAME_DT_SECONDS, MAX_FRAME_DT_SECONDS)

        if measurement is None:
            return self.predict(dt)

        if not self.initialized:
            self.initialize(*measurement)
            return self.position()

        self.predict(dt)
        return self.correct(*measurement)

    def position(self) -> tuple[float, float]:
        return float(self.state[0, 0]), float(self.state[1, 0])

    def velocity(self) -> tuple[float, float]:
        return float(self.state[2, 0]), float(self.state[3, 0])

    def project(self, dt: float) -> Optional[tuple[float, float]]:
        if not self.initialized:
            return None
        transition = self._transition_matrix(max(0.0, dt))
        future = transition @ self.state
        return float(future[0, 0]), float(future[1, 0])


class PIDController:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        integral_clamp: float = INTEGRAL_CLAMP,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_clamp = integral_clamp
        self.integral = 0.0
        self.previous_error: Optional[float] = None
        self.derivative = 0.0

    def compute(self, error: float, dt_scale: float) -> float:
        dt_scale = max(dt_scale, 1e-6)
        self.integral = clamp(
            self.integral + error * dt_scale,
            -self.integral_clamp,
            self.integral_clamp,
        )

        if self.previous_error is None:
            raw_derivative = 0.0
        else:
            raw_derivative = (error - self.previous_error) / dt_scale

        self.derivative = (
            DERIVATIVE_SMOOTHING * self.derivative
            + (1.0 - DERIVATIVE_SMOOTHING) * raw_derivative
        )
        self.previous_error = error

        return (
            self.kp * error
            + self.ki * self.integral
            + self.kd * self.derivative
        )

    def hold(self, reset_integral: bool = False) -> None:
        self.previous_error = None
        self.derivative = 0.0
        if reset_integral:
            self.integral = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = None
        self.derivative = 0.0


class ESP32SerialLink:
    def __init__(self, port: str, baud_rate: int):
        self.serial = serial.Serial(
            port,
            baud_rate,
            timeout=0.05,
            write_timeout=0.05,
            inter_byte_timeout=0.0,
        )
        time.sleep(2.0)
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()

    def send(self, pitch: float, yaw: float) -> None:
        pitch_byte = int(round(clamp(pitch, 0.0, 180.0)))
        yaw_byte = int(round(clamp(yaw, 0.0, 180.0)))
        # Keep only the freshest command; stale pan/tilt updates are worse than drops.
        self.serial.reset_output_buffer()
        self.serial.write(bytes([0xFF, pitch_byte, yaw_byte]))
        self.serial.flush()

    def send_immediate(self, pitch: float, yaw: float) -> None:
        try:
            self.serial.reset_output_buffer()
            self.serial.reset_input_buffer()
        except serial.SerialException:
            pass
        self.send(pitch, yaw)

    def close(self) -> None:
        if not self.serial.is_open:
            return
        try:
            self.serial.reset_output_buffer()
            self.serial.reset_input_buffer()
        except serial.SerialException:
            pass
        self.serial.close()


class YoloDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(source=frame, verbose=False)
        detections: list[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.detach().cpu().numpy()
            confidences = boxes.conf.detach().cpu().numpy()
            classes = (
                boxes.cls.detach().cpu().numpy().astype(int)
                if boxes.cls is not None
                else np.full(len(xyxy), -1, dtype=int)
            )
            names = result.names

            for index, (x1, y1, x2, y2) in enumerate(xyxy):
                confidence = float(confidences[index])
                if confidence < TARGET_CONFIDENCE_THRESHOLD:
                    continue

                class_id = int(classes[index]) if classes[index] >= 0 else None
                label = self._label_for(names, class_id)
                detections.append(
                    Detection(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=confidence,
                        class_id=class_id,
                        label=label,
                    )
                )

        return detections

    @staticmethod
    def _label_for(names: object, class_id: Optional[int]) -> str:
        if class_id is None:
            return "target"
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)


class PanTiltTracker:
    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_area = float(frame_width * frame_height)
        self.center_x = frame_width * 0.5
        self.center_y = frame_height * 0.5
        self.px_per_deg_x = frame_width / CAMERA_FOV_X
        self.px_per_deg_y = frame_height / CAMERA_FOV_Y
        self.deadband_x_deg = DEADBAND_PIXELS / max(self.px_per_deg_x, 1e-6)
        self.deadband_y_deg = DEADBAND_PIXELS / max(self.px_per_deg_y, 1e-6)

        self.current_pitch = clamp(90.0, PITCH_MIN, PITCH_MAX)
        self.current_yaw = clamp(90.0, YAW_MIN, YAW_MAX)
        self.state = "SWEEP"
        self.lost_frames = LOST_FRAMES_THRESHOLD
        self.sweep_angle = 0.0

        self.kalman = ConstantVelocityKalmanFilter(
            process_noise=KF_PROCESS_NOISE,
            measurement_noise=KF_MEAS_NOISE,
        )
        self.pid_yaw = PIDController(
            YAW_KP * self.px_per_deg_x,
            YAW_KI * self.px_per_deg_x,
            YAW_KD * self.px_per_deg_x,
        )
        self.pid_pitch = PIDController(
            PITCH_KP * self.px_per_deg_y,
            PITCH_KI * self.px_per_deg_y,
            PITCH_KD * self.px_per_deg_y,
        )

    def update(self, detections: Sequence[Detection], dt: float) -> TrackerTelemetry:
        dt = clamp(dt, MIN_FRAME_DT_SECONDS, MAX_FRAME_DT_SECONDS)
        target = self._select_target(detections)
        estimate: Optional[tuple[float, float]] = None
        aim_point: Optional[tuple[float, float]] = None
        error_x = 0.0
        error_y = 0.0

        if target is not None:
            if self.state != "TRACK":
                self.kalman.reset()
                self.pid_yaw.reset()
                self.pid_pitch.reset()
            self.state = "TRACK"
            self.lost_frames = 0
            estimate = self.kalman.update(dt, (target.center_x, target.center_y))
        elif self.state == "TRACK":
            self.lost_frames += 1
            estimate = self.kalman.update(dt, None)
            if self.lost_frames >= LOST_FRAMES_THRESHOLD or estimate is None:
                self._enter_sweep()
                estimate = None

        if self.state == "TRACK" and self.kalman.initialized:
            aim_point = self.kalman.project(PREDICTION_LEAD_SECONDS) or self.kalman.position()
            error_x = aim_point[0] - self.center_x
            error_y = aim_point[1] - self.center_y
            self._apply_tracking_control(error_x, error_y, dt)
        else:
            self._apply_sweep_control(dt)

        return TrackerTelemetry(
            state=self.state,
            detections=detections,
            target=target,
            estimate=estimate,
            aim_point=aim_point,
            error_x=error_x,
            error_y=error_y,
            current_pitch=self.current_pitch,
            current_yaw=self.current_yaw,
            lost_frames=self.lost_frames,
        )

    def _enter_sweep(self) -> None:
        self.state = "SWEEP"
        self.kalman.reset()
        self.pid_yaw.reset()
        self.pid_pitch.reset()

    def _select_target(self, detections: Sequence[Detection]) -> Optional[Detection]:
        if not detections:
            return None

        filtered = [
            detection
            for detection in detections
            if detection.area >= self.frame_area * MIN_BOX_AREA_RATIO
        ]
        if not filtered:
            filtered = list(detections)

        predicted = (
            self.kalman.project(PREDICTION_LEAD_SECONDS)
            if self.state == "TRACK" and self.kalman.initialized
            else None
        )
        if predicted is None:
            return max(filtered, key=lambda detection: (detection.confidence, detection.area))

        association_limit = max(self.frame_width, self.frame_height) * ASSOCIATION_DISTANCE_RATIO

        def score(detection: Detection) -> float:
            distance = math.hypot(
                detection.center_x - predicted[0],
                detection.center_y - predicted[1],
            )
            distance_score = max(0.0, 1.0 - (distance / max(association_limit, 1.0)))
            area_score = min(detection.area / max(self.frame_area, 1.0), 1.0)
            return (detection.confidence * 2.0) + distance_score + area_score

        best = max(filtered, key=score)
        best_distance = math.hypot(best.center_x - predicted[0], best.center_y - predicted[1])
        if best_distance > association_limit and best.confidence < 0.65:
            return max(filtered, key=lambda detection: (detection.confidence, detection.area))
        return best

    def _apply_tracking_control(self, error_x: float, error_y: float, dt: float) -> None:
        dt_scale = max(dt * CONTROL_REFERENCE_FPS, 1e-6)
        error_x_deg = error_x / max(self.px_per_deg_x, 1e-6)
        error_y_deg = error_y / max(self.px_per_deg_y, 1e-6)
        yaw_step = 0.0
        pitch_step = 0.0

        if abs(error_x_deg) > self.deadband_x_deg:
            yaw_step = self.pid_yaw.compute(error_x_deg, dt_scale)
            yaw_step = clamp(yaw_step, -MAX_YAW_SPEED_DPS * dt, MAX_YAW_SPEED_DPS * dt)
        else:
            self.pid_yaw.hold()

        if abs(error_y_deg) > self.deadband_y_deg:
            pitch_step = self.pid_pitch.compute(error_y_deg, dt_scale)
            pitch_step = clamp(pitch_step, -MAX_PITCH_SPEED_DPS * dt, MAX_PITCH_SPEED_DPS * dt)
        else:
            self.pid_pitch.hold()

        self.current_yaw = clamp(self.current_yaw - yaw_step, YAW_MIN, YAW_MAX)
        self.current_pitch = clamp(self.current_pitch + pitch_step, PITCH_MIN, PITCH_MAX)

    def _apply_sweep_control(self, dt: float) -> None:
        self.sweep_angle = (self.sweep_angle + SWEEP_SPEED * dt * CONTROL_REFERENCE_FPS) % (2.0 * math.pi)

        yaw_phase = (self.sweep_angle / math.pi) % 2.0
        triangle = yaw_phase if yaw_phase < 1.0 else 2.0 - yaw_phase
        target_yaw = YAW_MIN + triangle * (YAW_MAX - YAW_MIN)
        target_pitch = SWEEP_PITCH_CENTER + SWEEP_PITCH_AMP * math.sin(self.sweep_angle * SWEEP_PITCH_FREQ)

        self.current_pitch = self._move_toward(
            self.current_pitch,
            target_pitch,
            MAX_PITCH_SPEED_DPS,
            dt,
            PITCH_MIN,
            PITCH_MAX,
        )
        self.current_yaw = self._move_toward(
            self.current_yaw,
            target_yaw,
            MAX_YAW_SPEED_DPS,
            dt,
            YAW_MIN,
            YAW_MAX,
        )

    @staticmethod
    def _move_toward(
        current: float,
        target: float,
        max_speed_dps: float,
        dt: float,
        lower: float,
        upper: float,
    ) -> float:
        max_step = max_speed_dps * dt
        step = clamp(target - current, -max_step, max_step)
        return clamp(current + step, lower, upper)


def open_camera() -> cv2.VideoCapture:
    capture = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    if capture.isOpened():
        return capture

    capture.release()
    capture = cv2.VideoCapture(CAMERA_INDEX)
    if capture.isOpened():
        return capture

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

def main() -> None:
    detector = YoloDetector(MODEL_PATH)
    serial_link: Optional[ESP32SerialLink] = None
    capture: Optional[cv2.VideoCapture] = None
    tracker: Optional[PanTiltTracker] = None
    previous_time = time.monotonic()
    fps_ema = 0.0

    try:
        capture = open_camera()
        serial_link = ESP32SerialLink(SERIAL_PORT, BAUD_RATE)

        while True:
            ok, frame = capture.read()
            now = time.monotonic()
            dt = now - previous_time
            previous_time = now

            if not ok:
                raise RuntimeError("Camera read failed")

            if FRAME_ROTATION is not None:
                frame = cv2.rotate(frame, FRAME_ROTATION)

            if tracker is None:
                tracker = PanTiltTracker(frame.shape[1], frame.shape[0])

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
                serial_link.send_immediate(neutral_pitch, neutral_yaw)
        except Exception:
            pass
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()
        if serial_link is not None:
            serial_link.close()


if __name__ == "__main__":
    main()
