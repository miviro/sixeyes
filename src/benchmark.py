"""Benchmark detect and track latency over an entire input source."""

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from input_source import IMAGE_EXTS

_RUNS_DIR = Path(__file__).parent.parent / "runs"
IMGSZ = 1280  # must match SourceWorker's model.track() call
CONF = 0.6
WARMUP = 3


def _frames(spec: str):
    """Every frame of the source, unpaced (input_source plays files in real time)."""
    path = Path(spec.removeprefix("file:"))
    if path.is_dir():
        for p in sorted(path.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                img = cv2.imread(str(p))
                if img is not None:
                    yield img
        return
    cap = cv2.VideoCapture(int(spec[len("camera:"):]) if spec.startswith("camera:") else spec)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


def _measure(call, spec: str, name: str) -> list[float]:
    ms = []
    for frame in _frames(spec):
        t0 = time.perf_counter()
        call(frame)
        ms.append((time.perf_counter() - t0) * 1000)
        print(f"\r[bench] {name}: frame {len(ms)}", end="", flush=True)
    print()
    return ms


def _row(name: str, ms: list[float]) -> str:
    a = np.asarray(ms)
    return (f"{name:<8}{len(a):>8}{a.mean():>10.1f}{np.median(a):>10.1f}"
            f"{np.percentile(a, 95):>10.1f}{np.percentile(a, 99):>10.1f}"
            f"{1000 / a.mean():>8.1f}")


def run_benchmark(model_path: str, spec: str) -> None:
    model = YOLO(model_path, task="detect")

    def detect(f):
        model.predict(f, verbose=False, imgsz=IMGSZ, conf=CONF)

    def track(f):
        model.track(f, verbose=False, persist=True, imgsz=IMGSZ, conf=CONF)

    warm = next(_frames(spec), None)
    if warm is None:
        raise SystemExit(f"[bench] no frames read from {spec!r}")
    for _ in range(WARMUP):
        detect(warm)

    detect_ms = _measure(detect, spec, "detect")
    # track goes last: it attaches tracker callbacks to the predictor, which
    # would then also run on every later predict() call and skew it
    track_ms = _measure(track, spec, "track")

    report = "\n".join([
        f"model: {model_path}  |  input: {spec}  |  imgsz={IMGSZ} conf={CONF}",
        f"{'mode':<8}{'frames':>8}{'mean ms':>10}{'median':>10}{'p95':>10}{'p99':>10}{'fps':>8}",
        _row("detect", detect_ms),
        _row("track", track_ms),
    ])
    print(report)

    out = _RUNS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_benchmark.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report + "\n")
    print(f"[bench] saved → {out}")
