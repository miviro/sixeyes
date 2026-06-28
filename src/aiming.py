import math
import time
import serial
from config import HFOV_DEG, VFOV_DEG  # noqa: F401

# Servo limits and centres (must match esp32.ino)
YAW_CENTER   = 90.0
PITCH_CENTER = 70.0   # midpoint of PITCH_MIN=0 .. PITCH_MAX=140
YAW_MIN,   YAW_MAX   = 10.0, 170.0
PITCH_MIN, PITCH_MAX =  0.0, 140.0

# SG90 datasheet: 0.1 s / 60° at 4.8 V no-load → 600 °/s.
# Under typical load assume roughly half that.
SERVO_SPEED_DEG_S = 300.0

# How often to send a new aim command (seconds)
SEND_INTERVAL = 0.5

# Deadband: ignore target if it's within this many degrees of current estimated position
DEADBAND_DEG = 7.5

# Sweep: yaw triangle wave + pitch sine wave, time-driven
SWEEP_YAW_PERIOD   = 10.0  # seconds for one full yaw back-and-forth cycle
SWEEP_PITCH_AMP    = 20.0  # degrees either side of PITCH_CENTER
SWEEP_PITCH_CYCLES = 5     # pitch oscillations per yaw cycle

_ser: serial.Serial | None = None
_last_send = 0.0

# Last commanded angles
_cmd_yaw   = YAW_CENTER
_cmd_pitch = PITCH_CENTER

# Estimated current angles — updated every frame via update_estimated_angles()
_est_yaw   = YAW_CENTER
_est_pitch = PITCH_CENTER

# Sweep state
_sweep_start_t: float | None = None


def init_serial(port: str = "/dev/ttyUSB0", baud: int = 115200) -> None:
    global _ser
    _ser = serial.Serial(port, baud, timeout=0)


def update_estimated_angles(dt: float) -> tuple[float, float]:
    global _est_yaw, _est_pitch
    max_move = SERVO_SPEED_DEG_S * dt

    def step(est: float, cmd: float) -> float:
        diff = cmd - est
        return est + max(-max_move, min(max_move, diff))

    _est_yaw   = step(_est_yaw,   _cmd_yaw)
    _est_pitch = step(_est_pitch, _cmd_pitch)
    return _est_yaw, _est_pitch


def is_servo_moving() -> bool:
    return (abs(_est_yaw - _cmd_yaw) > 0.5 or
            abs(_est_pitch - _cmd_pitch) > 0.5)


def _send_raw(yaw: float, pitch: float) -> None:
    global _last_send, _cmd_yaw, _cmd_pitch
    yaw   = float(int(max(YAW_MIN,   min(YAW_MAX,   yaw))))
    pitch = float(int(max(PITCH_MIN, min(PITCH_MAX, pitch))))
    _cmd_yaw, _cmd_pitch = yaw, pitch
    _last_send = time.monotonic()
    if _ser and _ser.is_open:
        _ser.write(bytes([0xFF, int(pitch), int(yaw)]))
        _ser.flush()


def sweep_tick() -> None:
    global _sweep_start_t
    now = time.monotonic()
    if _sweep_start_t is None:
        _sweep_start_t = now
    t = now - _sweep_start_t

    phase    = (t % SWEEP_YAW_PERIOD) / SWEEP_YAW_PERIOD
    yaw_norm = 1.0 - abs(2.0 * phase - 1.0)
    yaw      = YAW_MIN + yaw_norm * (YAW_MAX - YAW_MIN)

    pitch_freq = SWEEP_PITCH_CYCLES / SWEEP_YAW_PERIOD
    pitch = PITCH_CENTER + SWEEP_PITCH_AMP * math.sin(2 * math.pi * pitch_freq * t)

    _send_raw(yaw, pitch)


def reset_sweep() -> None:
    global _sweep_start_t
    _sweep_start_t = None


def aim(world_yaw: float, world_pitch: float) -> None:
    global _last_send, _cmd_yaw, _cmd_pitch

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
        _ser.flush()
