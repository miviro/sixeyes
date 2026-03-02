import cv2
import math
import serial
import time
from ultralytics import YOLO

# --- Configuration ---
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE   = 115200
model       = YOLO("faces.pt")

# Servo travel limits (degrees)
YAW_MIN,   YAW_MAX   = 0,  140
PITCH_MIN, PITCH_MAX = 0,  180

# Deadband — pixel radius around centre treated as locked
DEADBAND_PIXELS = 100

# Frames without detection before falling back to sweep
LOST_FRAMES_THRESHOLD = 20

# Sweep: elliptical scan centred at (90°, 90°)
SWEEP_SPEED        = 0.03   # rad / frame
SWEEP_YAW_RADIUS   = 55.0   # degrees
SWEEP_PITCH_RADIUS = 35.0   # degrees

# PID gains  (output = degrees of servo movement per frame)
YAW_KP,   YAW_KI,   YAW_KD   = 0.01, 0.02, 0.0
PITCH_KP, PITCH_KI, PITCH_KD = 0.01, 0.02, 0.0
INTEGRAL_CLAMP = 20.0       # max absolute integral term (degrees)


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

current_pitch = 90.0
current_yaw   = 90.0

sweep_angle = 0.0
state       = "SWEEP"
lost_frames = 0

pid_yaw   = PIDController(YAW_KP,   YAW_KI,   YAW_KD)
pid_pitch = PIDController(PITCH_KP, PITCH_KI, PITCH_KD)


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
                    pid_yaw.reset()
                    pid_pitch.reset()
                    state = "TRACK"

                lost_frames = 0

        if not detected:
            lost_frames += 1
            if lost_frames >= LOST_FRAMES_THRESHOLD:
                if state == "TRACK":
                    pid_yaw.reset()
                    pid_pitch.reset()
                state = "SWEEP"

        if state == "TRACK" and face_x is not None:
            error_x = face_x - center_x
            error_y = face_y - center_y

            moved = False
            if abs(error_x) > DEADBAND_PIXELS:
                current_yaw -= pid_yaw.compute(error_x)
                moved = True

            if abs(error_y) > DEADBAND_PIXELS:
                current_pitch += pid_pitch.compute(error_y)
                moved = True

            if moved:
                current_yaw   = max(YAW_MIN,   min(YAW_MAX,   current_yaw))
                current_pitch = max(PITCH_MIN, min(PITCH_MAX, current_pitch))
                send_to_esp32(current_pitch, current_yaw)

        elif state == "SWEEP":
            sweep_angle   += SWEEP_SPEED
            current_yaw    = 90.0 + SWEEP_YAW_RADIUS   * math.cos(sweep_angle)
            current_pitch  = 90.0 + SWEEP_PITCH_RADIUS * math.sin(sweep_angle)
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
