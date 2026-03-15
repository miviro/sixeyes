import time
import cv2
from ultralytics import YOLO
from lstm import FRAME_W, FRAME_H, update_track, track_history
from drawing import draw_track, draw_hud
from aiming import init_serial, update_estimated_angles, aim

yolo = YOLO("faces.pt")
init_serial("/dev/ttyUSB0")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, 30)

prev_t = time.monotonic()

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    now   = time.monotonic()
    dt    = now - prev_t
    prev_t = now

    # Advance servo kinematic model — gives the angle the camera is actually
    # pointing at right now, accounting for SG90 slew time.
    servo_yaw, servo_pitch = update_estimated_angles(dt)

    results = yolo.track(frame, persist=True, verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        tids  = results[0].boxes.id.cpu().numpy().astype(int)

        for (cx, cy, w, h), tid in zip(boxes, tids):
            pred = update_track(tid, cx, cy, w, h, servo_yaw, servo_pitch)
            draw_track(frame, tid, cx, cy, w, h, track_history[tid], pred,
                       servo_yaw, servo_pitch)
            if pred is not None:
                aim(pred[0], pred[1])
                break  # aim at the first tracked face only

    draw_hud(frame)
    cv2.imshow("sixeyes", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
