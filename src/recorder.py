import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

_RUNS_DIR = Path(__file__).parent.parent / "runs"

_CSV_HEADER = ["detection_id", "frame", "track_id", "conf",
               "x1", "y1", "x2", "y2", "cx", "cy", "crop_file"]


class _SourceRecorder:
    def __init__(self, run_dir: Path, label: str):
        safe = label.replace("/", "_").replace(":", "_")
        src_dir = run_dir / safe
        src_dir.mkdir(parents=True, exist_ok=True)
        self._src_dir = src_dir

        self._crops_dir = src_dir / "crops"
        self._crops_dir.mkdir(exist_ok=True)

        self._raw_writer: cv2.VideoWriter | None = None
        self._ann_writer: cv2.VideoWriter | None = None

        self._csv_file = open(src_dir / "detections.csv", "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(_CSV_HEADER)

        self._det_id = 0

    def _init_writers(self, h: int, w: int) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._raw_writer = cv2.VideoWriter(
            str(self._src_dir / "raw.mp4"), fourcc, 30.0, (w, h)
        )
        self._ann_writer = cv2.VideoWriter(
            str(self._src_dir / "annotated.mp4"), fourcc, 30.0, (w, h)
        )

    def write(self, raw: np.ndarray, annotated: np.ndarray, results, frame_idx: int) -> None:
        h, w = raw.shape[:2]
        if self._raw_writer is None:
            self._init_writers(h, w)

        self._raw_writer.write(raw)
        self._ann_writer.write(annotated)

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return

        ids = (boxes.id.cpu().numpy().astype(int)
               if boxes.id is not None else [-1] * len(boxes))

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu())

            crop_name = f"{self._det_id:06d}.jpg"
            rx1 = max(0, int(x1))
            ry1 = max(0, int(y1))
            rx2 = min(w, int(x2))
            ry2 = min(h, int(y2))
            cv2.imwrite(str(self._crops_dir / crop_name), raw[ry1:ry2, rx1:rx2],
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

            self._csv.writerow([
                self._det_id, frame_idx, int(ids[i]), f"{conf:.3f}",
                f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                f"{(x1 + x2) / 2:.1f}", f"{(y1 + y2) / 2:.1f}",
                crop_name,
            ])
            self._det_id += 1

    def close(self) -> None:
        if self._raw_writer:
            self._raw_writer.release()
        if self._ann_writer:
            self._ann_writer.release()
        self._csv_file.close()


class Recorder:
    def __init__(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._dir = _RUNS_DIR / ts
        self._dir.mkdir(parents=True, exist_ok=True)
        self._sources: dict[str, _SourceRecorder] = {}
        print(f"[rec] saving to {self._dir}")

    def log(self, label: str, raw: np.ndarray, annotated: np.ndarray, results, frame_idx: int) -> None:
        if label not in self._sources:
            self._sources[label] = _SourceRecorder(self._dir, label)
        self._sources[label].write(raw, annotated, results, frame_idx)

    def close(self) -> None:
        for src in self._sources.values():
            src.close()
        print(f"\n[rec] {len(self._sources)} source(s) → {self._dir}")
