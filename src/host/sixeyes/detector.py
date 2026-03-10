from __future__ import annotations

from typing import Optional

import numpy as np
from ultralytics import YOLO

from .config import TARGET_CONFIDENCE_THRESHOLD, TARGET_LABEL
from .models import Detection


class YoloDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(source=frame, verbose=False)
        detections: list[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.detach().cpu().numpy()
            confidences = boxes.conf.detach().cpu().numpy()
            classes = (
                boxes.cls.detach().cpu().numpy().astype(int)
                if boxes.cls is not None
                else np.full(len(xyxy), -1, dtype=int)
            )
            names = result.names

            for index, (x1, y1, x2, y2) in enumerate(xyxy):
                confidence = float(confidences[index])
                if confidence < TARGET_CONFIDENCE_THRESHOLD:
                    continue

                class_id = int(classes[index]) if classes[index] >= 0 else None
                label = self._label_for(names, class_id)
                if label != TARGET_LABEL:
                    continue
                detections.append(
                    Detection(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=confidence,
                        class_id=class_id,
                        label=label,
                    )
                )

        return detections

    @staticmethod
    def _label_for(names: object, class_id: Optional[int]) -> str:
        if class_id is None:
            return "target"
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)
