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
lstm_enabled     = True
SWEEP_PATIENCE   = 30   # steps without a target before sweeping begins
_no_target_steps = 0
ignored_tids     = set()
current_tid      = None

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

    has_detections = results[0].boxes.id is not None
    active_box     = None  # first non-ignored detection

    if has_detections:
        boxes  = results[0].boxes.xywh.cpu().numpy()
        tids   = results[0].boxes.id.cpu().numpy().astype(int)
        moving = is_servo_moving()
        preds  = {}
        for (cx, cy, w, h), tid in zip(boxes, tids):
            ignored = tid in ignored_tids
            preds[tid] = update_track(tid, cx, cy, w, h, servo_yaw, servo_pitch,
                                      moving or not lstm_enabled or ignored)
            draw_track(frame, tid, cx, cy, w, h, track_history[tid],
                       preds[tid] if lstm_enabled and not ignored else None,
                       servo_yaw, servo_pitch)
            if active_box is None and not ignored:
                active_box = (cx, cy, w, h, tid)

    has_target = active_box is not None
    if has_target:
        cx, cy, w, h, tid = active_box
        current_tid = tid
        _no_target_steps = 0
        if _no_target_steps >= SWEEP_PATIENCE:
            # Came back from a real sweep — treat everything as fresh
            ignored_tids.clear()
        reset_sweep()
        if lstm_enabled:
            if preds[tid] is not None:
                aim(preds[tid][0], preds[tid][1])
        else:
            det_yaw   = servo_yaw   + (0.5 - cx / FRAME_W) * HFOV_DEG
            det_pitch = servo_pitch + (cy / FRAME_H - 0.5)  * VFOV_DEG
            aim(det_yaw, det_pitch)
    else:
        current_tid = None
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
    elif key == ord("n") and current_tid is not None:
        ignored_tids.add(current_tid)
        current_tid = None

cap.release()
cv2.destroyAllWindows()
