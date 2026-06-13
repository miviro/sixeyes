import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO
from input_source import open_input
from display import make_grid
from config import MOG_DELTA_FRAMES, MOG_VAR_THRESHOLD, MOG_DETECT_SHADOWS, MOG_SCALE_W, MOG_SCALE_H

if len(sys.argv) < 3:
    sys.exit("usage: sixeyes.py <model.pt> <camera:N | video | folder>")

model = YOLO(sys.argv[1])
model_label = sys.argv[1].rsplit("/", 1)[-1].rsplit(".", 1)[0]

mog = cv2.createBackgroundSubtractorMOG2(
    history=MOG_DELTA_FRAMES,
    varThreshold=MOG_VAR_THRESHOLD,
    detectShadows=MOG_DETECT_SHADOWS,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX

for frame in open_input(sys.argv[2]):
    t0 = time.perf_counter()
    small   = cv2.resize(frame, (MOG_SCALE_W, MOG_SCALE_H), interpolation=cv2.INTER_LINEAR)
    fg_mask = mog.apply(small)
    fg_mask = cv2.resize(fg_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    fg_bgr  = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    t1 = time.perf_counter()

    results  = model.track(frame, verbose=False, persist=True, imgsz=(1920, 1088), half=True)
    yolo_bgr = results[0].plot()
    t2 = time.perf_counter()

    panels = [
        ("Raw",         frame),
        ("MoG BG",      fg_bgr),
        (model_label,   yolo_bgr),
    ]
    grid = make_grid(panels)
    t3 = time.perf_counter()

    mog_ms   = (t1 - t0) * 1000
    yolo_ms  = (t2 - t1) * 1000
    grid_ms  = (t3 - t2) * 1000
    total_ms = (t3 - t0) * 1000

    bar_h = 28
    bar = np.zeros((bar_h, grid.shape[1], 3), dtype=np.uint8)
    label = (f"MoG {mog_ms:.1f}ms  |  YOLO {yolo_ms:.1f}ms  |"
             f"  Grid {grid_ms:.1f}ms  |  Total {total_ms:.1f}ms  ({1000/total_ms:.1f} fps)")
    cv2.putText(bar, label, (8, bar_h - 8), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    print(f"\n{label}", end="", flush=True)
    cv2.imshow("sixeyes", np.vstack([grid, bar]))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
