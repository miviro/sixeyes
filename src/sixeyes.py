import sys
import os
import glob
import time
import cv2
from ultralytics import YOLO
from lstm import FRAME_W, FRAME_H, HFOV_DEG, VFOV_DEG, update_track, track_history
from drawing import draw_track, draw_hud, draw_sweep_hud
from aiming import init_serial, update_estimated_angles, aim, is_servo_moving, sweep_tick, reset_sweep

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ---------------------------------------------------------------------------
# Arguments:  <model>[:<label>]  <input>  [--track]
#
#   model   name of the .pt file without extension (e.g. faces, yolo11m)
#   label   class name to filter detections (e.g. yolo11m:bottle)
#   input   camera:<n>  |  path to video file  |  path to image folder
#   --track move servos to follow the target
# ---------------------------------------------------------------------------

def _usage():
    sys.exit(
        "usage: sixeyes.py <model>[:<label>] <camera:N | video | folder> [--track]"
    )

raw_args   = sys.argv[1:]
tracking   = "--track" in raw_args
pos_args   = [a for a in raw_args if a != "--track"]

if len(pos_args) < 2:
    _usage()

# --- model ---
model_str = pos_args[0]
if ":" in model_str:
    model_name, filter_label = model_str.split(":", 1)
else:
    model_name, filter_label = model_str, None

# --- input ---
raw_input = pos_args[1]

if raw_input.startswith("camera:"):
    input_type = "camera"
    cap_source = int(raw_input[len("camera:"):])
else:
    path = raw_input[len("file:"):] if raw_input.startswith("file:") else raw_input
    if os.path.isdir(path):
        input_type = "folder"
    else:
        input_type = "video"
    cap_source = path

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

yolo = YOLO(f"{model_name}")

if filter_label is not None:
    names_inv = {v: k for k, v in yolo.names.items()}
    if filter_label not in names_inv:
        sys.exit(f"Label {filter_label!r} not in model. Available: {sorted(names_inv)}")
    track_classes = [names_inv[filter_label]]
else:
    track_classes = None

if tracking:
    init_serial("/dev/ttyUSB0")

# --- frame source ---
if input_type == "folder":
    img_paths = sorted(
        p for p in glob.glob(os.path.join(cap_source, "*"))
        if os.path.splitext(p)[1].lower() in IMAGE_EXTS
    )
    if not img_paths:
        sys.exit(f"No images found in {cap_source!r}")
    out_path = cap_source.rstrip("/\\") + "_out.mp4"
    cap = None
else:
    cap = cv2.VideoCapture(cap_source)
    out_path = None
    if input_type == "camera":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_FPS, 30)

# target ms per frame for real-time pacing (camera is naturally paced)
if input_type == "camera":
    frame_ms = 1
elif input_type == "video":
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_ms = int(1000 / fps)
else:  # folder
    frame_ms = 33  # 30 fps

def _frames():
    if input_type == "folder":
        for p in img_paths:
            img = cv2.imread(p)
            if img is not None:
                yield img
    else:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            yield frame

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

prev_t           = time.monotonic()
lstm_enabled     = True
SWEEP_PATIENCE   = 30
_no_target_steps = 0
ignored_tids     = set()
current_tid      = None
writer           = None
last_frame       = None
quit_requested   = False

for frame in _frames():
    now    = time.monotonic()
    dt     = now - prev_t
    prev_t = now

    servo_yaw, servo_pitch = update_estimated_angles(dt) if tracking else (0.0, 0.0)

    results = yolo.track(frame, persist=True, verbose=False, classes=track_classes)

    has_detections = results[0].boxes.id is not None
    active_box     = None

    if has_detections:
        boxes  = results[0].boxes.xywh.cpu().numpy()
        tids   = results[0].boxes.id.cpu().numpy().astype(int)
        moving = is_servo_moving() if tracking else False
        preds  = {}
        for (cx, cy, w, h), tid in zip(boxes, tids):
            ignored    = tid in ignored_tids
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
        current_tid       = tid
        _no_target_steps  = 0
        if tracking:
            reset_sweep()
            if lstm_enabled and preds[tid] is not None:
                aim(preds[tid][0], preds[tid][1])
            else:
                det_yaw   = servo_yaw   + (0.5 - cx / FRAME_W) * HFOV_DEG
                det_pitch = servo_pitch + (cy / FRAME_H - 0.5)  * VFOV_DEG
                aim(det_yaw, det_pitch)
    else:
        current_tid       = None
        _no_target_steps += 1
        if tracking and _no_target_steps >= SWEEP_PATIENCE:
            sweep_tick()

    sweeping = tracking and not has_target and _no_target_steps >= SWEEP_PATIENCE
    draw_hud(frame, sweeping=sweeping, lstm_on=lstm_enabled,
             patience=max(0, SWEEP_PATIENCE - _no_target_steps) if not has_target else 0)

    if input_type == "folder":
        if writer is None:
            fh, fw = frame.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (fw, fh))
        writer.write(frame)

    last_frame = frame
    cv2.imshow("sixeyes", frame)
    elapsed_ms = int((time.monotonic() - now) * 1000)
    key = cv2.waitKey(max(1, frame_ms - elapsed_ms)) & 0xFF
    if key == ord("q"):
        quit_requested = True
        break
    elif key == ord("l"):
        lstm_enabled = not lstm_enabled
    elif key == ord("n") and current_tid is not None:
        ignored_tids.add(current_tid)
        current_tid = None

if not quit_requested and last_frame is not None:
    while cv2.waitKey(100) & 0xFF != ord("q"):
        pass

if writer:
    writer.release()
    print(f"saved {out_path}")
if cap:
    cap.release()
cv2.destroyAllWindows()
