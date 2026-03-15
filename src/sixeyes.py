import time
import cv2
from ultralytics import YOLO
from lstm import FRAME_W, FRAME_H, HFOV_DEG, VFOV_DEG, update_track, track_history
from drawing import draw_track, draw_hud, draw_sweep_hud
from aiming import init_serial, update_estimated_angles, aim, is_servo_moving, sweep_tick, reset_sweep

yolo = YOLO("faces.pt")
init_serial("/dev/ttyUSB0")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, 30)

prev_t = time.monotonic()
lstm_enabled  = True
SWEEP_PATIENCE = 30   # steps without a target before sweeping begins
_no_target_steps = 0

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

    has_target = results[0].boxes.id is not None
    if has_target:
        _no_target_steps = 0
        boxes = results[0].boxes.xywh.cpu().numpy()
        tids  = results[0].boxes.id.cpu().numpy().astype(int)

        reset_sweep()
        moving = is_servo_moving()
        for (cx, cy, w, h), tid in zip(boxes, tids):
            # When LSTM is off, pass servo_moving=True to skip training/history
            pred = update_track(tid, cx, cy, w, h, servo_yaw, servo_pitch,
                                moving or not lstm_enabled)
            draw_track(frame, tid, cx, cy, w, h, track_history[tid],
                       pred if lstm_enabled else None,
                       servo_yaw, servo_pitch)
            if lstm_enabled:
                if pred is not None:
                    aim(pred[0], pred[1])
            else:
                det_yaw   = servo_yaw   + (0.5 - cx / FRAME_W) * HFOV_DEG
                det_pitch = servo_pitch + (cy / FRAME_H - 0.5)  * VFOV_DEG
                aim(det_yaw, det_pitch)
            break  # aim at the first tracked face only
    else:
        _no_target_steps += 1
        if _no_target_steps >= SWEEP_PATIENCE:
            sweep_tick()

    sweeping = not has_target and _no_target_steps >= SWEEP_PATIENCE
    draw_hud(frame, sweeping=sweeping, lstm_on=lstm_enabled,
             patience=max(0, SWEEP_PATIENCE - _no_target_steps) if not has_target else 0)
    cv2.imshow("sixeyes", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("l"):
        lstm_enabled = not lstm_enabled

cap.release()
cv2.destroyAllWindows()
