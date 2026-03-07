import cv2
import math
import serial
import time
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE   = 115200
model       = YOLO("faces.pt")

# Servo travel limits (degrees)
YAW_MIN,   YAW_MAX   = 0,   160
PITCH_MIN, PITCH_MAX = 10,  140

# Deadband — pixel radius around centre treated as locked
DEADBAND_PIXELS = 20

# Frames without detection before falling back to sweep
LOST_FRAMES_THRESHOLD = 20

# Sweep: pitch 0-180 triangle wave, yaw sinusoid ±20° around centre
SWEEP_SPEED      = 0.05   # rad / frame — pitch advance rate (~4s per full 0→180→0 at 30fps)
SWEEP_YAW_CENTER = (YAW_MIN + YAW_MAX) / 2   # 80°
SWEEP_YAW_AMP    = 20.0   # ±degrees of yaw oscillation
SWEEP_YAW_FREQ   = 6      # yaw oscillations per pitch triangle period

# PID gains  (output = degrees of servo movement per frame)
YAW_KP,   YAW_KI,   YAW_KD   = 0.01, 0.0, 0.02
PITCH_KP, PITCH_KI, PITCH_KD = 0.01, 0.0, 0.02
INTEGRAL_CLAMP = 20.0       # max absolute integral term (degrees)

# Kalman filter noise (pixels²)
KF_PROCESS_NOISE = 5.0
KF_MEAS_NOISE    = 5.0

# Pixels-per-degree calibration for motion feed-forward.
# After 90° CCW rotation: x-axis ↔ yaw, y-axis ↔ pitch.
# Rough default: frame_pixels / lens_FOV_degrees. Tune empirically.
CAMERA_FOV_X = 62.0   # degrees along rotated-frame x (yaw axis)
CAMERA_FOV_Y = 48.0   # degrees along rotated-frame y (pitch axis)


class KalmanFilter2D:
    """
    Constant-velocity Kalman filter for a 2-D point.
    State : [x, y, vx, vy]   Measurement : [x, y]
    """
    def __init__(self):
        dt = 1.0
        self.kf = cv2.KalmanFilter(4, 2, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        self.kf.controlMatrix = np.array([
            [1, 0],
            [0, 1],
            [0, 0],
            [0, 0],
        ], dtype=np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * KF_PROCESS_NOISE
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KF_MEAS_NOISE
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self.initialized = False

    def step(self, x=None, y=None, control_dx=0.0, control_dy=0.0):
        """
        One Kalman cycle per frame.
        With (x, y): predict + correct → posterior estimate.
        Without    : predict only      → extrapolated estimate.
        Returns (est_x, est_y) or (None, None) before first detection.
        """
        if not self.initialized:
            if x is None:
                return None, None
            self.kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
            self.initialized  = True

        control = np.array([[control_dx], [control_dy]], dtype=np.float32)
        prior = self.kf.predict(control)
        if x is not None:
            post = self.kf.correct(np.array([[x], [y]], dtype=np.float32))
            return float(post[0]), float(post[1])
        return float(prior[0]), float(prior[1])

    def reset(self):
        self.initialized = False
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)


class PIDController:
    """Single-axis PID controller. Output is servo angle delta (degrees)."""
    def __init__(self, kp: float, ki: float, kd: float,
                 integral_clamp: float = INTEGRAL_CLAMP):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_clamp = integral_clamp
        self._integral   = 0.0
        self._prev_error = 0.0

    def compute(self, error: float) -> float:
        self._integral = max(-self.integral_clamp,
                             min(self.integral_clamp, self._integral + error))
        derivative       = error - self._prev_error
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0


# --- Hardware / model init ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)
except Exception as e:
    print(f"Serial Error: {e}")
    exit()

cap    = cv2.VideoCapture(0, cv2.CAP_V4L2)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# After 90° CCW rotation, frame dimensions swap
center_x = height // 2
center_y = width  // 2
px_per_deg_x = height / CAMERA_FOV_X   # rotated frame x spans 'height' px
px_per_deg_y = width  / CAMERA_FOV_Y   # rotated frame y spans 'width'  px

current_pitch = 90.0
current_yaw   = 90.0

sweep_angle = 0.0
state       = "SWEEP"
lost_frames = 0

pid_yaw   = PIDController(YAW_KP,   YAW_KI,   YAW_KD)
pid_pitch = PIDController(PITCH_KP, PITCH_KI, PITCH_KD)
kalman    = KalmanFilter2D()

pending_dx = 0.0
pending_dy = 0.0


def send_to_esp32(p, y):
    ser.write(bytes([0xFF, int(p), int(y)]))


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame           = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        annotated_frame = frame.copy()
        results         = model.track(frame, persist=True, stream=True)

        detected = False
        face_x, face_y = None, None

        for r in results:
            annotated_frame = r.plot()
            if r.boxes and r.boxes.id is not None:
                detected = True
                box      = r.boxes.xywh[0].cpu().numpy()
                face_x, face_y = float(box[0]), float(box[1])

                if state == "SWEEP":
                    kalman.reset()
                    pid_yaw.reset()
                    pid_pitch.reset()
                    state = "TRACK"

                lost_frames = 0

        if not detected:
            lost_frames += 1
            if lost_frames >= LOST_FRAMES_THRESHOLD:
                if state == "TRACK":
                    kalman.reset()
                    pid_yaw.reset()
                    pid_pitch.reset()
                state = "SWEEP"

        est_x, est_y = kalman.step(face_x, face_y,
                                   control_dx=pending_dx, control_dy=pending_dy)
        pending_dx = 0.0
        pending_dy = 0.0

        if state == "TRACK" and est_x is not None:
            error_x = est_x - center_x
            error_y = est_y - center_y

            moved = False
            if abs(error_x) > DEADBAND_PIXELS:
                delta_yaw    = pid_yaw.compute(error_x)
                current_yaw += delta_yaw
                pending_dx   = -delta_yaw * px_per_deg_x
                moved = True

            if abs(error_y) > DEADBAND_PIXELS:
                delta_pitch    = pid_pitch.compute(error_y)
                current_pitch += delta_pitch
                pending_dy     = -delta_pitch * px_per_deg_y
                moved = True

            if moved:
                current_yaw   = max(YAW_MIN,   min(YAW_MAX,   current_yaw))
                current_pitch = max(PITCH_MIN, min(PITCH_MAX, current_pitch))
                send_to_esp32(current_pitch, current_yaw)

        elif state == "SWEEP":
            sweep_angle  += SWEEP_SPEED
            # Pitch: triangle wave PITCH_MIN → PITCH_MAX → PITCH_MIN → ...
            pitch_phase   = (sweep_angle / math.pi) % 2
            t             = pitch_phase if pitch_phase < 1 else 2 - pitch_phase
            current_pitch = PITCH_MIN + t * (PITCH_MAX - PITCH_MIN)
            # Yaw: sinusoid ±SWEEP_YAW_AMP around centre, faster than pitch
            current_yaw   = SWEEP_YAW_CENTER + SWEEP_YAW_AMP * math.sin(sweep_angle * SWEEP_YAW_FREQ)
            send_to_esp32(current_pitch, current_yaw)

        color = (0, 255, 0) if state == "TRACK" else (0, 165, 255)
        cv2.putText(annotated_frame, f"State: {state}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(annotated_frame, f"Yaw: {current_yaw:.1f}  Pitch: {current_pitch:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Face Tracker", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    ser.reset_output_buffer()
    send_to_esp32(90, 90)
    time.sleep(0.15)
    cap.release()
    cv2.destroyAllWindows()
    ser.close()
