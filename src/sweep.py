import math
import serial
import time

SERIAL_PORT  = "/dev/ttyUSB0"
BAUD_RATE    = 115200

# Arduino hard limits: 20-160 for both axes → radius = 70
YAW_CENTER    = 90.0
PITCH_CENTER  = 90.0
YAW_RADIUS    = 70.0   # 90 ± 70 → 20–160
PITCH_RADIUS  = 70.0   # 90 ± 70 → 20–160

SPEED_START   = 0.01   # radians per iteration (slow)
SPEED_MAX     = 0.08   # radians per iteration (fast)
SPEED_INC     = 0.0001 # added each iteration
LOOP_DELAY    = 0.02   # seconds per iteration (~50Hz)

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)
    print("Connected. Sweeping... Ctrl-C to stop.")
except Exception as e:
    print(f"Serial Error: {e}")
    exit()

angle = 0.0
speed = SPEED_START

try:
    while True:
        angle += speed
        speed  = min(speed + SPEED_INC, SPEED_MAX)

        yaw   = YAW_CENTER   + YAW_RADIUS   * math.cos(angle)
        pitch = PITCH_CENTER + PITCH_RADIUS  * math.sin(angle)

        ser.write(f"{int(pitch)},{int(yaw)}\n".encode())
        while ser.in_waiting > 0:
            ser.readline()

        time.sleep(LOOP_DELAY)
except KeyboardInterrupt:
    print("Stopped.")
finally:
    ser.close()
