import argparse
import signal
import threading
import time
from dataclasses import dataclass
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
parser.add_argument("--headless", action="store_true", help="run without display windows")
parser.add_argument("--port", default=FOLLOW_PORT, help="serial port for --follow")
args = parser.parse_args()

model_label = Path(args.model).stem

if args.follow:
    aiming.init_serial(args.port)
    print(f"[follow] serial on {args.port}")


@dataclass(frozen=True)
class Snapshot:
    seq: int
    annotated: np.ndarray
    last_bb: np.ndarray | None
    target: tuple[float, float] | None   # (tx, ty) of best detection
    size: tuple[int, int]                # (fw, fh)
    dz: tuple[int, int, int, int]        # dead zone (x1, y1, x2, y2)
    infer_ms: float


class SourceWorker:
    """Owns a capture buffer and a YOLO instance; runs inference in its own thread."""

    def __init__(self, spec: str, label: str, model_path: str, recorder: Recorder):
        self.label = label
        self.buf = FrameBuffer(spec)
        self._model_path = model_path
        self._recorder = recorder
        self._lock = threading.Lock()
        self._snapshot: Snapshot | None = None
        threading.Thread(target=self._run, daemon=True, name=f"worker-{label}").start()

    def snapshot(self) -> Snapshot | None:
        with self._lock:
            return self._snapshot

    def _run(self):
        model = YOLO(self._model_path, task="detect")
        last_seq = 0
        last_bb: np.ndarray | None = None
        frame_idx = 0

        while True:
            seq, frame = self.buf.wait(last_seq)
            if frame is None or seq <= last_seq:
                if self.buf.finished:
                    break
                continue
            last_seq = seq

            t0 = time.perf_counter()
            results = model.track(frame, verbose=False, persist=True, imgsz=640, conf=0.6)
            infer_ms = (time.perf_counter() - t0) * 1000

            fh, fw = frame.shape[:2]
            margin = (1.0 - FOLLOW_ZONE) / 2.0
            dz = (int(fw * margin), int(fh * margin),
                  int(fw * (1 - margin)), int(fh * (1 - margin)))

            annotated = results[0].plot()
            cv2.rectangle(annotated, dz[:2], dz[2:], DZ_COLOR, 2)

            target = None
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                best = int(boxes.conf.cpu().numpy().argmax())
                x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy()
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(fw, int(x2)), min(fh, int(y2))
                target = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                last_bb = frame[y1:y2, x1:x2].copy()

            snap = Snapshot(seq, annotated, last_bb, target, (fw, fh), dz, infer_ms)
            with self._lock:
                self._snapshot = snap

            self._recorder.log(self.label, frame, annotated, results[0], frame_idx)
            frame_idx += 1


# ---- Source registry (main thread only) ----
recorder = Recorder()
workers: dict[str, SourceWorker] = {}
monitor: EighteeyesMonitor | None = None


def _add_source(key: str, spec: str, label: str):
    if key in workers:
        return
    workers[key] = SourceWorker(spec, label, args.model, recorder)
    print(f"[+] {label}")


for spec in args.sources:
    if spec == "eighteyes":
        if monitor is None:
            monitor = EighteeyesMonitor()
            print("Scanning eighteyes1..16.local in background...")
    else:
        _add_source(spec, spec, spec)

# ---- Main loop: display + aiming ----
FONT = cv2.FONT_HERSHEY_SIMPLEX
FRAME_BUDGET = 1.0 / 30.0
ema_ms: float | None = None
frame_count = 0
no_target_frames = 0
prev_t = time.perf_counter()
last_consumed: dict[str, int] = {}

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

        panels = []
        active_labels = []
        follow_status = ""
        found_target = False
        n_connected = sum(1 for w in workers.values() if w.buf.get()[1] is not None)

        for key, w in workers.items():
            _, frame = w.buf.get()
            if frame is None:
                continue

            suffix = f" [{w.label}]" if n_connected > 1 else ""
            snap = w.snapshot()
            if w.buf.finished:
                suffix += " (finished)"
                active_labels.append(f"{w.label} (finished)")
            else:
                active_labels.append(
                    f"{w.label} {snap.infer_ms:.0f}ms" if snap else w.label
                )

            panels.append((f"Raw{suffix}", frame))
            bb_panel = (snap.last_bb if snap and snap.last_bb is not None
                        else np.zeros((1, 1, 3), dtype=np.uint8))
            panels.append((f"Last Det{suffix}", bb_panel))

            if snap is None:
                continue
            panels.append((f"{model_label}{suffix}", snap.annotated))

            # Aiming: only act on inference results not yet consumed
            if snap.seq <= last_consumed.get(key, 0):
                continue
            last_consumed[key] = snap.seq

            if snap.target is not None:
                found_target = True
                if args.follow:
                    no_target_frames = 0
                    aiming.reset_sweep()
                    tx, ty = snap.target
                    fw, fh = snap.size
                    dz_x1, dz_y1, dz_x2, dz_y2 = snap.dz
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
            time.sleep(0.05)
            continue

        loop_ms = dt * 1000
        frame_count += 1
        ema_ms = loop_ms if ema_ms is None else ema_ms + EMA_ALPHA * (loop_ms - ema_ms)

        src_str = "  +  ".join(active_labels) if active_labels else "no sources"
        status = (f"{1000 / ema_ms:.1f} fps  |  "
                  f"frame {frame_count}  |  {src_str}{follow_status}")

        print(f"\r{status}", end="", flush=True)

        if args.headless:
            if workers and all(w.buf.finished for w in workers.values()):
                break
            time.sleep(max(0.001, FRAME_BUDGET - (time.perf_counter() - t0)))
            continue

        grid = make_grid(panels)
        bar = np.zeros((28, grid.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, status, (8, 20), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow("sixeyes", np.vstack([grid, bar]))

        # Pace the display loop at ~30 fps; waitKey doubles as the sleep
        delay_ms = max(1, int((FRAME_BUDGET - (time.perf_counter() - t0)) * 1000))
        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break
finally:
    # A second Ctrl-C during cleanup would abort recorder.close() and leave
    # the writer thread wedged at interpreter shutdown
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if not args.headless:
        cv2.destroyAllWindows()
    recorder.close()
