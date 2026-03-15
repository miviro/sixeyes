import time
import serial
from lstm import HFOV_DEG, VFOV_DEG  # noqa: F401 — re-exported for drawing.py

# Servo limits and centres (must match esp32.ino)
YAW_CENTER   = 90.0
PITCH_CENTER = 70.0   # midpoint of PITCH_MIN=0 .. PITCH_MAX=140
YAW_MIN,   YAW_MAX   = 10.0, 170.0
PITCH_MIN, PITCH_MAX =  0.0, 140.0

# SG90 datasheet: 0.1 s / 60° at 4.8 V no-load → 600 °/s.
# Under typical load assume roughly half that.
SERVO_SPEED_DEG_S = 300.0

# How often to send a new aim command (seconds)
SEND_INTERVAL = 1.0

# Deadband: ignore target if it's within this many degrees of current estimated position
DEADBAND_DEG = 10.0

_ser: serial.Serial | None = None
_last_send = 0.0

# Last commanded angles
_cmd_yaw   = YAW_CENTER
_cmd_pitch = PITCH_CENTER

# Estimated current angles — updated every frame via update_estimated_angles()
_est_yaw   = YAW_CENTER
_est_pitch = PITCH_CENTER


def init_serial(port: str = "/dev/ttyUSB0", baud: int = 115200) -> None:
    global _ser
    _ser = serial.Serial(port, baud, timeout=0)


def update_estimated_angles(dt: float) -> tuple[float, float]:
    """Advance the servo position estimate by dt seconds toward the commanded angles.

    Returns the estimated (yaw, pitch) the camera is currently pointing at.
    Call this once per frame before using the angles for world-space conversion.
    """
    global _est_yaw, _est_pitch
    max_move = SERVO_SPEED_DEG_S * dt

    def step(est: float, cmd: float) -> float:
        diff = cmd - est
        return est + max(-max_move, min(max_move, diff))

    _est_yaw   = step(_est_yaw,   _cmd_yaw)
    _est_pitch = step(_est_pitch, _cmd_pitch)
    return _est_yaw, _est_pitch


def aim(world_yaw: float, world_pitch: float) -> None:
    """Send servo frame for the predicted world-space target position.

    world_yaw / world_pitch come directly from the LSTM prediction, which
    already operates in world-space angles — no pixel conversion needed.
    """
    global _last_send, _cmd_yaw, _cmd_pitch

    # Skip if target is within the deadband around current estimated position
    if (abs(world_yaw - _est_yaw) <= DEADBAND_DEG and
            abs(world_pitch - _est_pitch) <= DEADBAND_DEG):
        return

    now = time.monotonic()
    if now - _last_send < SEND_INTERVAL:
        return
    _last_send = now

    yaw   = int(max(YAW_MIN,   min(YAW_MAX,   world_yaw)))
    pitch = int(max(PITCH_MIN, min(PITCH_MAX, world_pitch)))

    _cmd_yaw   = float(yaw)
    _cmd_pitch = float(pitch)

    if _ser and _ser.is_open:
        _ser.write(bytes([0xFF, pitch, yaw]))
