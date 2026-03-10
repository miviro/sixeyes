from __future__ import annotations

import math
from typing import Optional, Sequence

from .config import (
    ASSOCIATION_DISTANCE_RATIO,
    CAMERA_FOV_X,
    CAMERA_FOV_Y,
    CONTROL_REFERENCE_FPS,
    DEADBAND_PIXELS,
    KF_MEAS_NOISE,
    KF_PROCESS_NOISE,
    LOST_FRAMES_THRESHOLD,
    MAX_PITCH_SPEED_DPS,
    MAX_YAW_SPEED_DPS,
    MIN_BOX_AREA_RATIO,
    PITCH_KD,
    PITCH_KI,
    PITCH_KP,
    PITCH_MAX,
    PITCH_MIN,
    PREDICTION_LEAD_SECONDS,
    SWEEP_PITCH_AMP,
    SWEEP_PITCH_CENTER,
    SWEEP_PITCH_FREQ,
    SWEEP_SPEED,
    YAW_KD,
    YAW_KI,
    YAW_KP,
    YAW_MAX,
    YAW_MIN,
    clamp,
)
from .kalman import ConstantVelocityKalmanFilter
from .models import Detection, TrackerTelemetry
from .pid import PIDController


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
        self._prev_yaw = self.current_yaw
        self._prev_pitch = self.current_pitch
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
        # Ego-motion compensation: convert servo delta (degrees) to expected
        # pixel displacement of a stationary target in the frame.
        # Sign convention: increasing yaw rotates the camera so targets shift
        # in the -x direction; increasing pitch shifts targets in the -y direction.
        delta_yaw = self.current_yaw - self._prev_yaw
        delta_pitch = self.current_pitch - self._prev_pitch
        if self.state == "TRACK":
            ego_dx = delta_yaw * self.px_per_deg_x
            ego_dy = -delta_pitch * self.px_per_deg_y
        else:
            ego_dx = ego_dy = 0.0
        self._prev_yaw = self.current_yaw
        self._prev_pitch = self.current_pitch

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
            estimate = self.kalman.update(dt, (target.center_x, target.center_y), ego_dx, ego_dy)
        elif self.state == "TRACK":
            self.lost_frames += 1
            estimate = self.kalman.update(dt, None, ego_dx, ego_dy)
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

        filtered = [d for d in detections if d.area >= self.frame_area * MIN_BOX_AREA_RATIO] or list(detections)

        predicted = (
            self.kalman.project(PREDICTION_LEAD_SECONDS)
            if self.state == "TRACK" and self.kalman.initialized
            else None
        )
        if predicted is None:
            return max(filtered, key=lambda d: (d.confidence, d.area))

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
            return max(filtered, key=lambda d: (d.confidence, d.area))
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

        # Yaw: triangle wave across the full servo range
        yaw_phase = (self.sweep_angle / math.pi) % 2.0
        triangle = yaw_phase if yaw_phase < 1.0 else 2.0 - yaw_phase
        target_yaw = YAW_MIN + triangle * (YAW_MAX - YAW_MIN)

        # Pitch: sinusoid around centre
        target_pitch = SWEEP_PITCH_CENTER + SWEEP_PITCH_AMP * math.sin(self.sweep_angle * SWEEP_PITCH_FREQ)

        self.current_pitch = self._move_toward(
            self.current_pitch, target_pitch, MAX_PITCH_SPEED_DPS, dt, PITCH_MIN, PITCH_MAX,
        )
        self.current_yaw = self._move_toward(
            self.current_yaw, target_yaw, MAX_YAW_SPEED_DPS, dt, YAW_MIN, YAW_MAX,
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
