from __future__ import annotations

from typing import Optional

import numpy as np

from .config import MAX_FRAME_DT_SECONDS, MIN_FRAME_DT_SECONDS, clamp


class ConstantVelocityKalmanFilter:
    def __init__(self, process_noise: float, measurement_noise: float):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state = np.zeros((4, 1), dtype=np.float64)
        self.covariance = np.eye(4, dtype=np.float64)
        self.initialized = False

        # Cache constant matrices — H (measurement) and R (measurement noise)
        self._H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)
        self._R = np.eye(2, dtype=np.float64) * measurement_noise

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

    def predict(self, dt: float, dx: float = 0.0, dy: float = 0.0) -> Optional[tuple[float, float]]:
        if not self.initialized:
            return None

        transition = self._transition_matrix(dt)
        self.state = transition @ self.state
        # Compensate for known camera ego-motion so the filter does not mistake
        # pixel displacement caused by servo movement for target motion.
        self.state[0, 0] += dx
        self.state[1, 0] += dy
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
        innovation = measurement - self._H @ self.state
        innovation_covariance = self._H @ self.covariance @ self._H.T + self._R
        kalman_gain = self.covariance @ self._H.T @ np.linalg.inv(innovation_covariance)

        identity = np.eye(4, dtype=np.float64)
        self.state = self.state + kalman_gain @ innovation
        correction = identity - kalman_gain @ self._H
        self.covariance = (
            correction @ self.covariance @ correction.T
            + kalman_gain @ self._R @ kalman_gain.T
        )
        return self.position()

    def update(
        self,
        dt: float,
        measurement: Optional[tuple[float, float]],
        dx: float = 0.0,
        dy: float = 0.0,
    ) -> Optional[tuple[float, float]]:
        dt = clamp(dt, MIN_FRAME_DT_SECONDS, MAX_FRAME_DT_SECONDS)

        if measurement is None:
            return self.predict(dt, dx, dy)

        self.predict(dt, dx, dy)  # no-op when not yet initialized
        return self.correct(*measurement)

    def position(self) -> tuple[float, float]:
        return float(self.state[0, 0]), float(self.state[1, 0])

    def project(self, dt: float) -> Optional[tuple[float, float]]:
        if not self.initialized:
            return None
        transition = self._transition_matrix(max(0.0, dt))
        future = transition @ self.state
        return float(future[0, 0]), float(future[1, 0])
