import os
import glob
import threading
import time
import cv2
from config import CAMERA_W, CAMERA_H, CAMERA_FPS, EIGHTEYES_MAX_INDEX

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def open_input(raw_input: str):
    if raw_input.startswith("camera:"):
        return _camera(int(raw_input[len("camera:"):]))
    if raw_input.startswith("http://") or raw_input.startswith("https://"):
        return _stream(raw_input)
    path = raw_input[len("file:"):] if raw_input.startswith("file:") else raw_input
    if os.path.isdir(path):
        return _folder(path)
    return _video(path)


class EighteeyesMonitor:
    """Background thread that probes eighteyes1..N.local via DNS only (no TCP to port 81)."""

    def __init__(self, max_index: int = EIGHTEYES_MAX_INDEX, interval: float = 5.0):
        self._max_index = max_index
        self._interval = interval
        self._active: dict[int, str] = {}
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True, name="eighteyes-monitor").start()

    @staticmethod
    def _probe(n: int):
        import socket
        try:
            infos = socket.getaddrinfo(f"eighteyes{n}.local", None, socket.AF_INET)
            ip = infos[0][4][0]
            return (n, f"http://{ip}:81/")
        except socket.gaierror:
            return None

    def _run(self):
        import concurrent.futures
        while True:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_index) as ex:
                results = list(ex.map(self._probe, range(1, self._max_index + 1)))
            with self._lock:
                self._active = dict(r for r in results if r)
            time.sleep(self._interval)

    def get_active(self) -> dict[int, str]:
        with self._lock:
            return dict(self._active)


def _stream(url: str):
    while True:
        cap = cv2.VideoCapture(url)
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()
        time.sleep(1)


def _camera(index: int):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def _video(path: str):
    cap = cv2.VideoCapture(path)
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def _folder(path: str):
    img_paths = sorted(
        p for p in glob.glob(os.path.join(path, "*"))
        if os.path.splitext(p)[1].lower() in IMAGE_EXTS
    )
    for p in img_paths:
        img = cv2.imread(p)
        if img is not None:
            yield img
