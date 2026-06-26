import os
import sys
import time
import threading
import cv2
import numpy as np
from ultralytics import YOLO
from input_source import open_input, EighteeyesMonitor
from display import make_grid
from config import MOG_DELTA_FRAMES, MOG_VAR_THRESHOLD, MOG_DETECT_SHADOWS, MOG_SCALE_W, MOG_SCALE_H

if len(sys.argv) < 3:
    sys.exit("usage: sixeyes.py <model.pt> <source1> [source2 ...]")

model_path = sys.argv[1]
model_label = model_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]


class _FrameBuffer:
    """Reads a source in a background thread; always holds the latest frame."""

    def __init__(self, spec: str):
        self._frame = None
        self._lock = threading.Lock()
        threading.Thread(target=self._run, args=(spec,), daemon=True).start()

    def _run(self, spec: str):
        for frame in open_input(spec):
            with self._lock:
                self._frame = frame

    def get(self):
        with self._lock:
            return self._frame


def _make_mog():
    return cv2.createBackgroundSubtractorMOG2(
        history=MOG_DELTA_FRAMES,
        varThreshold=MOG_VAR_THRESHOLD,
        detectShadows=MOG_DETECT_SHADOWS,
    )


# Registry: key -> (_FrameBuffer, mog, YOLO | None, display_label)
_registry: dict[str, tuple] = {}
_registry_lock = threading.Lock()


def _load_model_bg(key: str):
    m = YOLO(model_path, task="detect")
    with _registry_lock:
        if key in _registry:
            buf, mog, _, label = _registry[key]
            _registry[key] = (buf, mog, m, label)


def _add_source(key: str, spec: str, label: str):
    with _registry_lock:
        if key in _registry:
            return
        _registry[key] = (_FrameBuffer(spec), _make_mog(), None, label)
    threading.Thread(target=_load_model_bg, args=(key,), daemon=True).start()
    print(f"[+] {label}")


# --- Parse args ---
monitor: EighteeyesMonitor | None = None
for _spec in sys.argv[2:]:
    if _spec == "eighteyes":
        if monitor is None:
            monitor = EighteeyesMonitor()
            print("Scanning eighteyes1..16.local in the background...")
    else:
        _add_source(_spec, _spec, _spec)


def _sync_monitor():
    """Add a source the first time DNS resolves a device. Never removes."""
    if monitor is None:
        return
    active = monitor.get_active()
    with _registry_lock:
        current_keys = {k for k in _registry if k.startswith("eighteyes")}
    for n, url in active.items():
        key = f"eighteyes{n}"
        if key not in current_keys:
            _add_source(key, url, f"eighteyes{n}")


FONT = cv2.FONT_HERSHEY_SIMPLEX
EMA_ALPHA = 0.05
ema_total = None
frame_count = 0

while True:
    t0 = time.perf_counter()

    _sync_monitor()

    with _registry_lock:
        entries = list(_registry.items())

    panels = []
    active_labels = []
    n_connected = sum(1 for _, (b, *_) in entries if b.get() is not None)

    for key, (buf, mog, model, label) in entries:
        frame = buf.get()
        if frame is None:
            continue  # not yet connected

        small = cv2.resize(frame, (MOG_SCALE_W, MOG_SCALE_H), interpolation=cv2.INTER_LINEAR)
        fg_mask = mog.apply(small)
        fg_mask = cv2.resize(fg_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        fg_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        suffix = f" [{label}]" if n_connected > 1 else ""
        panels.append((f"Raw{suffix}", frame))
        panels.append((f"MoG{suffix}", fg_bgr))
        active_labels.append(label)

        if model is not None:
            results = model.track(frame, verbose=False, persist=True, imgsz=128, conf=0.1)
            yolo_bgr = results[0].plot()
            panels.append((f"{model_label}{suffix}", yolo_bgr))

    if not panels:
        time.sleep(0.01)
        continue

    total_ms = (time.perf_counter() - t0) * 1000
    frame_count += 1
    if ema_total is None:
        ema_total = total_ms
    else:
        ema_total += EMA_ALPHA * (total_ms - ema_total)

    grid = make_grid(panels)

    bar_h = 28
    bar = np.zeros((bar_h, grid.shape[1], 3), dtype=np.uint8)
    src_str = "  +  ".join(active_labels) if active_labels else "no sources"
    status = (f"Total {ema_total:.1f}ms  ({1000 / ema_total:.1f} fps avg)  |  "
              f"frame {frame_count}  |  {src_str}")
    cv2.putText(bar, status, (8, bar_h - 8), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    print(f"\r{status}", end="", flush=True)
    cv2.imshow("sixeyes", np.vstack([grid, bar]))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
