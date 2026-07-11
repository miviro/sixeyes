import queue
import threading
import time

import cv2
import numpy as np
import requests

from config import ALERT_COOLDOWN_S, NTFY_SERVER


class Notifier:
    """Pushes a ntfy alert with the detection crop attached whenever a new
    track id appears; a single sender thread does all HTTP so inference
    never blocks on the network."""

    def __init__(self, topic: str, max_queue: int = 8):
        self._url = f"{NTFY_SERVER.rstrip('/')}/{topic}"
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._lock = threading.Lock()
        self._seen: dict[str, set[int]] = {}
        self._last_sent = -ALERT_COOLDOWN_S
        threading.Thread(target=self._run, daemon=True, name="notifier").start()
        print(f"[ntfy] alerts to {self._url}")

    def alert(self, label: str, frame: np.ndarray, results) -> None:
        boxes = results.boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return
        ids = boxes.id.cpu().numpy().astype(int)

        with self._lock:
            seen = self._seen.setdefault(label, set())
            new = [i for i in range(len(ids)) if int(ids[i]) not in seen]
            if not new:
                return
            # ids seen during cooldown are consumed without alerting
            seen.update(int(ids[i]) for i in new)
            now = time.monotonic()
            if now - self._last_sent < ALERT_COOLDOWN_S:
                return
            self._last_sent = now

        conf = boxes.conf.cpu().numpy()
        best = max(new, key=lambda i: conf[i])
        tid = int(ids[best])
        cls_name = results.names[int(boxes.cls[best])]

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy()
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        crop = frame[y1:y2, x1:x2].copy()

        extra = f" (+{len(new) - 1})" if len(new) > 1 else ""
        message = f"{cls_name} nuevo (track {tid}, conf {conf[best]:.2f}){extra}"
        try:
            self._queue.put_nowait((label, tid, message, crop))
        except queue.Full:
            pass

    def _run(self) -> None:
        while True:
            label, tid, message, crop = self._queue.get()
            params = {"title": f"sixeyes: {label}", "message": message,
                      "priority": "high"}
            body = b""
            headers = {}
            if crop.size > 0:
                ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    body = jpg.tobytes()
                    headers["Filename"] = f"track{tid}.jpg"
            try:
                requests.post(self._url, data=body, params=params,
                              headers=headers, timeout=10)
            except requests.RequestException as e:
                print(f"\n[ntfy] send failed: {e}")
