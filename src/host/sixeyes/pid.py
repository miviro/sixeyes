from __future__ import annotations

from typing import Optional

from .config import DERIVATIVE_SMOOTHING, INTEGRAL_CLAMP, clamp


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
