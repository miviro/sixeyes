import cv2
import serial
import time
from ultralytics import YOLO

# --- Configuration ---
SERIAL_PORT = "/dev/ttyUSB0"  # Adjust to your identified port
BAUD_RATE = 115200
# Load standard YOLOv11 face model or your custom weight
model = YOLO("yolo11m.pt") 

# Open Serial Connection
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2) # Allow ESP32 to reset
except Exception as e:
    print(f"Serial Error: {e}")
    exit()

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Target center coordinates
center_x = width // 2
center_y = height // 2

# Initial Servo Positions
current_pitch = 90
current_yaw = 90

def send_to_esp32(p, y):
    cmd = f"{int(p)},{int(y)}\n"
    ser.write(cmd.encode())

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run tracking
    results = model.track(frame, persist=True, stream=True, classes=[0]) # Class 0 is usually person/face

    for r in results:
        annotated_frame = r.plot()
        
        # Check if any boxes were detected
        if r.boxes and r.boxes.id is not None:
            # Get the first detected face (index 0)
            # box.xywh returns [center_x, center_y, width, height]
            box = r.boxes.xywh[0].cpu().numpy()
            target_x, target_y = box[0], box[1]

            # Calculate error from center (-1.0 to 1.0 scale)
            error_x = (target_x - center_x) / center_x
            error_y = (target_y - center_y) / center_y

            # Update servo logic: 
            # If target is to the right (positive error), increase yaw
            # If target is above (negative error), decrease pitch
            current_yaw -= error_x * 5  # Gain factor: adjust 5 for speed
            current_pitch += error_y * 5

            # Keep values within physical limits
            current_yaw = max(0, min(180, current_yaw))
            current_pitch = max(0, min(180, current_pitch))

            send_to_esp32(current_pitch, current_yaw)

        cv2.imshow('Face Tracker', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
