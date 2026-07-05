import threading
import time
import concurrent.futures
import socket
import cv2
from pathlib import Path
from config import CAMERA_W, CAMERA_H, CAMERA_FPS, EIGHTEYES_MAX_INDEX

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class FrameBuffer:
    """Reads a source in a background thread; always holds the latest frame."""

    def __init__(self, spec: str):
        self._frame = None
        self._seq = 0
        self.finished = False
        self._cond = threading.Condition()
        threading.Thread(target=self._run, args=(spec,), daemon=True).start()

    def _run(self, spec: str):
        for frame in open_input(spec):
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()
        with self._cond:
            self.finished = True
            self._cond.notify_all()

    def get(self):
        with self._cond:
            return self._seq, self._frame

    def wait(self, last_seq: int, timeout: float = 0.5):
        """Block until a frame newer than last_seq arrives (or timeout)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > last_seq, timeout=timeout)
            return self._seq, self._frame


class EighteeyesMonitor:
    """Background thread that probes eighteyes1..N.local via DNS every few seconds."""

    def __init__(self, max_index: int = EIGHTEYES_MAX_INDEX, interval: float = 5.0):
        self._max_index = max_index
        self._interval = interval
        self._active: dict[int, str] = {}
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True, name="eighteyes-monitor").start()

    @staticmethod
    def _probe(n: int):
        try:
            ip = socket.getaddrinfo(f"eighteyes{n}.local", None, socket.AF_INET)[0][4][0]
            return (n, f"http://{ip}:81/")
        except socket.gaierror:
            return None

    def _run(self):
        while True:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_index) as ex:
                results = list(ex.map(self._probe, range(1, self._max_index + 1)))
            with self._lock:
                self._active = dict(r for r in results if r)
            time.sleep(self._interval)

    def get_active(self) -> dict[int, str]:
        with self._lock:
            return dict(self._active)


def open_input(raw_input: str):
    if raw_input.startswith("camera:"):
        return _camera(int(raw_input[len("camera:"):]))
    if raw_input.startswith(("http://", "https://")):
        return _stream(raw_input)
    path = Path(raw_input.removeprefix("file:"))
    if path.is_dir():
        return _folder(path)
    return _video(str(path))


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
    while True:
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
        time.sleep(1)


def _paced(frames, fps: float):
    """Yield frames at the given rate so file sources play in real time."""
    interval = 1.0 / fps
    next_t = time.perf_counter()
    for frame in frames:
        now = time.perf_counter()
        if now < next_t:
            time.sleep(next_t - now)
        next_t = max(next_t + interval, now)
        yield frame


def _video(path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    def frames():
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()

    return _paced(frames(), fps)


def _folder(path: Path):
    def frames():
        for p in sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            img = cv2.imread(str(p))
            if img is not None:
                yield img

    return _paced(frames(), 30.0)
