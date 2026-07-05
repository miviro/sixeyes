import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import aiming
from config import (FOLLOW_PORT, FOLLOW_ZONE, HFOV_DEG, VFOV_DEG,
                    SWEEP_PATIENCE, EMA_ALPHA, DZ_COLOR)
from display import make_grid
from input_source import FrameBuffer, EighteeyesMonitor
from recorder import Recorder

# ---- CLI ----
parser = argparse.ArgumentParser(prog="sixeyes")
parser.add_argument("model", help="YOLO model .pt file")
parser.add_argument("sources", nargs="+",
                    help="input sources: camera:N, http://..., path/to/file, path/to/dir, eighteyes")
parser.add_argument("--follow", action="store_true", help="enable servo tracking via serial")
parser.add_argument("--port", default=FOLLOW_PORT, help="serial port for --follow")
args = parser.parse_args()

model_label = Path(args.model).stem

if args.follow:
    aiming.init_serial(args.port)
    print(f"[follow] serial on {args.port}")

# ---- Source registry ----
_registry: dict[str, tuple] = {}
_registry_lock = threading.Lock()
monitor: EighteeyesMonitor | None = None


def _load_model_bg(key: str):
    m = YOLO(args.model, task="detect")
    with _registry_lock:
        if key in _registry:
            buf, last_bb, _, label = _registry[key]
            _registry[key] = (buf, last_bb, m, label)


def _add_source(key: str, spec: str, label: str):
    with _registry_lock:
        if key in _registry:
            return
        _registry[key] = (FrameBuffer(spec), None, None, label)
    threading.Thread(target=_load_model_bg, args=(key,), daemon=True).start()
    print(f"[+] {label}")


for spec in args.sources:
    if spec == "eighteyes":
        if monitor is None:
            monitor = EighteeyesMonitor()
            print("Scanning eighteyes1..16.local in background...")
    else:
        _add_source(spec, spec, spec)

# ---- Main loop ----
FONT = cv2.FONT_HERSHEY_SIMPLEX
ema_ms: float | None = None
frame_count = 0
no_target_frames = 0
prev_t = time.perf_counter()
recorder = Recorder()

try:
    while True:
        t0 = time.perf_counter()
        dt = t0 - prev_t
        prev_t = t0

        servo_yaw, servo_pitch = (
            aiming.update_estimated_angles(dt) if args.follow
            else (aiming.YAW_CENTER, aiming.PITCH_CENTER)
        )

        # Sync newly discovered eighteyes cameras
        if monitor:
            for n, url in monitor.get_active().items():
                _add_source(f"eighteyes{n}", url, f"eighteyes{n}")

        with _registry_lock:
            entries = list(_registry.items())

        panels = []
        active_labels = []
        follow_status = ""
        found_target = False
        n_connected = sum(1 for _, (b, *_) in entries if b.get() is not None)

        for key, (buf, last_bb, model, label) in entries:
            frame = buf.get()
            if frame is None:
                continue

            fh, fw = frame.shape[:2]
            suffix = f" [{label}]" if n_connected > 1 else ""
            active_labels.append(label)

            # Dead zone: centre FOLLOW_ZONE fraction of the frame
            margin = (1.0 - FOLLOW_ZONE) / 2.0
            dz_x1, dz_y1 = int(fw * margin), int(fh * margin)
            dz_x2, dz_y2 = int(fw * (1 - margin)), int(fh * (1 - margin))

            panels.append((f"Raw{suffix}", frame))
            bb_panel = last_bb if last_bb is not None else np.zeros((1, 1, 3), dtype=np.uint8)
            panels.append((f"Last Det{suffix}", bb_panel))

            if model is not None:
                results = model.track(frame, verbose=False, persist=True, imgsz=640, conf=0.6)
                yolo_bgr = results[0].plot()
                cv2.rectangle(yolo_bgr, (dz_x1, dz_y1), (dz_x2, dz_y2), DZ_COLOR, 2)
                panels.append((f"{model_label}{suffix}", yolo_bgr))
                recorder.log(label, frame, yolo_bgr, results[0], frame_count)

                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    best = int(boxes.conf.cpu().numpy().argmax())
                    x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy()
                    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(fw, int(x2)), min(fh, int(y2))
                    tx, ty = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                    new_bb = frame[y1:y2, x1:x2].copy()
                    with _registry_lock:
                        if key in _registry:
                            buf2, _, m2, lbl2 = _registry[key]
                            _registry[key] = (buf2, new_bb, m2, lbl2)

                    found_target = True
                    if args.follow:
                        no_target_frames = 0
                        aiming.reset_sweep()
                        if dz_x1 <= tx <= dz_x2 and dz_y1 <= ty <= dz_y2:
                            follow_status = f"  |  follow: in zone ({tx:.0f},{ty:.0f})"
                        else:
                            world_yaw   = servo_yaw   + (0.5 - tx / fw) * HFOV_DEG
                            world_pitch = servo_pitch + (ty / fh - 0.5) * VFOV_DEG
                            aiming.aim(world_yaw, world_pitch)
                            follow_status = f"  |  follow: → {world_yaw:.1f}° {world_pitch:.1f}°"
                elif args.follow:
                    follow_status = "  |  follow: no det"

        if args.follow and not found_target:
            no_target_frames += 1
            if no_target_frames >= SWEEP_PATIENCE:
                aiming.sweep_tick()
                follow_status = f"  |  follow: sweeping ({no_target_frames}f)"

        if not panels:
            time.sleep(0.01)
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000
        frame_count += 1
        ema_ms = elapsed_ms if ema_ms is None else ema_ms + EMA_ALPHA * (elapsed_ms - ema_ms)

        src_str = "  +  ".join(active_labels) if active_labels else "no sources"
        status = (f"Total {ema_ms:.1f}ms  ({1000 / ema_ms:.1f} fps avg)  |  "
                  f"frame {frame_count}  |  {src_str}{follow_status}")

        grid = make_grid(panels)
        bar = np.zeros((28, grid.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, status, (8, 20), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        print(f"\r{status}", end="", flush=True)
        cv2.imshow("sixeyes", np.vstack([grid, bar]))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cv2.destroyAllWindows()
    recorder.close()
