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

# Proportional gain applied to the angular pixel error each frame.
# 1.0 = move the full error in one step (servo physical speed is the only limit).
# 0.5 = close half the remaining error per frame — smoother, slightly laggier.
KP = 0.5

# Ignore errors smaller than this (degrees) — mechanical noise floor of the SG90.
DEADBAND_DEG = 2.0

_ser: serial.Serial | None = None

# Float commands — fractional degrees are preserved between frames.
# Only cast to int at the point of the serial write.
_cmd_yaw   = YAW_CENTER
_cmd_pitch = PITCH_CENTER

# Estimated current angles — updated every frame via update_estimated_angles().
# Used only for world-space LSTM visualisation, NOT for the P-controller target.
_est_yaw   = YAW_CENTER
_est_pitch = PITCH_CENTER

# True when the latest face was inside the deadband (no command sent)
locked = True


def init_serial(port: str = "/dev/ttyUSB0", baud: int = 115200) -> None:
    global _ser
    _ser = serial.Serial(port, baud, timeout=0)


def update_estimated_angles(dt: float) -> tuple[float, float]:
    """Advance the servo position estimate by dt seconds toward the commanded angles.

    Returns estimated (yaw, pitch). Used only for world-space conversion in the
    LSTM visualisation — the P-controller drives from pixel error, not this estimate.
    """
    global _est_yaw, _est_pitch
    max_move = SERVO_SPEED_DEG_S * dt

    def step(est: float, cmd: float) -> float:
        diff = cmd - est
        return est + max(-max_move, min(max_move, diff))

    _est_yaw   = step(_est_yaw,   _cmd_yaw)
    _est_pitch = step(_est_pitch, _cmd_pitch)
    return _est_yaw, _est_pitch


def aim(cx_norm: float, cy_norm: float) -> None:
    """P-controller step driven purely from pixel error — no servo estimate involved.

    cx_norm / cy_norm: normalised face position in frame [0, 1].
    Converts pixel offset from centre to angular error using the camera FOV,
    then nudges the servo command by KP * error each frame.
    Float commands are accumulated internally; only truncated to int on serial write
    so sub-degree corrections are not discarded between frames.
    """
    global _cmd_yaw, _cmd_pitch, locked

    # Angular distance the face is from the frame centre
    error_yaw   = (cx_norm - 0.5) * HFOV_DEG   # positive → face right of centre
    error_pitch = (cy_norm - 0.5) * VFOV_DEG   # positive → face below centre

    if abs(error_yaw) < DEADBAND_DEG and abs(error_pitch) < DEADBAND_DEG:
        locked = True
        return

    locked = False

    # Yaw is flipped: face right of centre → pan right → decrease yaw
    _cmd_yaw   -= KP * error_yaw
    _cmd_pitch += KP * error_pitch

    # Clamp — keep as float to preserve fractional degrees
    _cmd_yaw   = max(YAW_MIN,   min(YAW_MAX,   _cmd_yaw))
    _cmd_pitch = max(PITCH_MIN, min(PITCH_MAX,  _cmd_pitch))

    if _ser and _ser.is_open:
        _ser.write(bytes([0xFF, int(_cmd_pitch), int(_cmd_yaw)]))
