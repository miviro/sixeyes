import cv2
import math
import serial
import time
from ultralytics import YOLO

# --- Configuration ---
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
model = YOLO("faces.pt")  # face-specific model; download from Ultralytics hub

# Servo travel limits (degrees)
YAW_MIN,   YAW_MAX   = 30,  150
PITCH_MIN, PITCH_MAX = 50,  130

# Tracking: proportional gain and pixel deadband
GAIN            = 3.0   # degrees moved per unit of normalised error
DEADBAND_PIXELS = 10    # pixel radius around centre treated as "locked"

# How many consecutive frames without a detection before switching to sweep
LOST_FRAMES_THRESHOLD = 20

# Sweep: circular scan centred at (90°, 90°)
SWEEP_SPEED        = 0.03   # radians per frame (~7s per full circle at 30fps)
SWEEP_YAW_RADIUS   = 55.0   # degrees — fits within 30-150 yaw range
SWEEP_PITCH_RADIUS = 35.0   # degrees — fits within 50-130 pitch range

# Open Serial Connection
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)  # Allow ESP32 to reset
except Exception as e:
    print(f"Serial Error: {e}")
    exit()

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# After 90° CCW rotation, dimensions are swapped
center_x = height // 2
center_y = width  // 2

# Servo state
current_pitch = 90.0
current_yaw   = 90.0

sweep_angle = 0.0        # current angle around the circle (radians)
state       = "SWEEP"    # "SWEEP" | "TRACK"
lost_frames = 0


def send_to_esp32(p, y):
    ser.write(bytes([0xFF, int(p), int(y)]))


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Correct for camera mounted 90° clockwise
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        annotated_frame = frame.copy()
        results = model.track(frame, persist=True, stream=True)

        detected = False
        for r in results:
            annotated_frame = r.plot()

            if r.boxes and r.boxes.id is not None:
                detected    = True
                lost_frames = 0
                state       = "TRACK"

                box = r.boxes.xywh[0].cpu().numpy()
                face_x, face_y = box[0], box[1]

                error_x = face_x - center_x
                error_y = face_y - center_y

                moved = False
                if abs(error_x) > DEADBAND_PIXELS:
                    current_yaw -= (error_x / center_x) * GAIN
                    moved = True
                if abs(error_y) > DEADBAND_PIXELS:
                    current_pitch += (error_y / center_y) * GAIN
                    moved = True

                if moved:
                    current_yaw   = max(YAW_MIN,   min(YAW_MAX,   current_yaw))
                    current_pitch = max(PITCH_MIN, min(PITCH_MAX, current_pitch))
                    send_to_esp32(current_pitch, current_yaw)

        # Count frames without a detection and fall back to sweep when stale
        if not detected:
            lost_frames += 1
            if lost_frames >= LOST_FRAMES_THRESHOLD:
                state = "SWEEP"

        # Sweep: circular scan across both axes
        if state == "SWEEP":
            sweep_angle   += SWEEP_SPEED
            current_yaw    = 90.0 + SWEEP_YAW_RADIUS   * math.cos(sweep_angle)
            current_pitch  = 90.0 + SWEEP_PITCH_RADIUS  * math.sin(sweep_angle)
            send_to_esp32(current_pitch, current_yaw)

        # Overlay: state and current servo position
        color = (0, 255, 0) if state == "TRACK" else (0, 165, 255)
        cv2.putText(annotated_frame, f"State: {state}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(annotated_frame, f"Yaw: {current_yaw:.1f}  Pitch: {current_pitch:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Face Tracker", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    ser.reset_output_buffer()         # discard any frames still queued in the OS TX buffer
    send_to_esp32(90, 90)             # center before PWM cuts
    time.sleep(0.15)                  # let Arduino idle-timeout fire
    cap.release()
    cv2.destroyAllWindows()
    ser.close()
