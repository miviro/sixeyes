import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

_RUNS_DIR = Path(__file__).parent.parent / "runs"


class Recorder:
    def __init__(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._dir = _RUNS_DIR / ts
        self._dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = self._dir / "detections.csv"
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow([
            "timestamp", "frame", "source", "track_id",
            "class_id", "class_name", "conf",
            "x1", "y1", "x2", "y2", "cx", "cy",
        ])

        self._videos: dict[str, cv2.VideoWriter] = {}
        print(f"[rec] saving to {self._dir}")

    def log(self, label: str, annotated: np.ndarray, results, frame_idx: int) -> None:
        # --- annotated video (one file per source) ---
        if label not in self._videos:
            h, w = annotated.shape[:2]
            safe = label.replace("/", "_").replace(":", "_")
            self._videos[label] = cv2.VideoWriter(
                str(self._dir / f"{safe}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0, (w, h),
            )
        self._videos[label].write(annotated)

        # --- CSV rows (one per detection) ---
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return

        ts = f"{time.time():.3f}"
        names = results.names
        ids = (boxes.id.cpu().numpy().astype(int)
               if boxes.id is not None else [-1] * len(boxes))

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            cls_id = int(boxes.cls[i].cpu())
            self._csv.writerow([
                ts, frame_idx, label,
                int(ids[i]), cls_id, names.get(cls_id, ""),
                f"{float(boxes.conf[i].cpu()):.3f}",
                f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                f"{(x1 + x2) / 2:.1f}", f"{(y1 + y2) / 2:.1f}",
            ])

    def close(self) -> None:
        for v in self._videos.values():
            v.release()
        self._csv_file.close()
        size_kb = self._csv_path.stat().st_size // 1024
        print(f"\n[rec] {len(self._videos)} video(s) + detections.csv ({size_kb} KB) → {self._dir}")
