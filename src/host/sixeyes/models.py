from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


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
