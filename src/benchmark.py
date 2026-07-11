"""Model speed benchmark: measures inference latency on the current hardware.

Results are written to runs/<timestamp>_benchmark/ as JSON (raw data, including
per-iteration latencies for plotting) and Markdown (human-readable summary).
"""

import json
import platform
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
import ultralytics
from ultralytics import YOLO

from config import CAMERA_W, CAMERA_H

_RUNS_DIR = Path(__file__).parent.parent / "runs"

WARMUP = 5
PIPELINE_IMGSZ = 1280  # must match SourceWorker's model.track() call
PIPELINE_CONF = 0.6
SWEEP_IMGSZ = [320, 640, 960, 1280]


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _device_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "cpu": _cpu_name(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "cuda": torch.cuda.is_available(),
    }
    if info["cuda"]:
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["vram_gb"] = round(props.total_memory / 2**30, 1)
    info["device"] = info.get("gpu", info["cpu"])
    return info


def _grab_frame(sources) -> tuple[np.ndarray, str]:
    """First frame of the first usable source, or a synthetic 1080p frame."""
    from input_source import open_input

    for spec in sources or []:
        if spec == "eighteyes":
            continue  # needs network discovery; not worth blocking a benchmark
        holder = {}

        def _read(spec=spec):
            try:
                holder["frame"] = next(iter(open_input(spec)))
            except StopIteration:
                pass

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout=5.0)
        if "frame" in holder:
            return holder["frame"], spec
        print(f"[bench] could not read a frame from {spec!r}, skipping")

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
    return frame, f"synthetic {CAMERA_W}x{CAMERA_H} noise"


def _stats(lat: list[float]) -> dict:
    s = sorted(lat)

    def pct(p: float) -> float:
        return s[min(len(s) - 1, int(len(s) * p))]

    return {
        "n": len(lat),
        "mean_ms": statistics.fmean(lat),
        "median_ms": statistics.median(lat),
        "std_ms": statistics.stdev(lat) if len(lat) > 1 else 0.0,
        "min_ms": s[0],
        "max_ms": s[-1],
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "fps": 1000.0 / statistics.fmean(lat),
    }


def _timed_run(call, frame: np.ndarray, n: int) -> dict:
    proc = psutil.Process()
    lat: list[float] = []
    stages = {"preprocess": [], "inference": [], "postprocess": []}
    psutil.cpu_percent()  # reset the system-wide counter
    rss_before = proc.memory_info().rss

    for _ in range(n):
        t0 = time.perf_counter()
        r = call(frame)
        lat.append((time.perf_counter() - t0) * 1000)
        for k in stages:
            stages[k].append(r[0].speed[k])

    out = _stats(lat)
    out["stages_mean_ms"] = {k: statistics.fmean(v) for k, v in stages.items()}
    out["cpu_percent"] = psutil.cpu_percent()  # avg over the loop, all cores
    out["rss_mb"] = proc.memory_info().rss / 2**20
    out["rss_delta_mb"] = (proc.memory_info().rss - rss_before) / 2**20
    out["latencies_ms"] = [round(v, 3) for v in lat]
    return out


def _fmt_row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |\n"


def _report(d: dict) -> str:
    info = d["device"]
    md = f"# SixEyes benchmark — {Path(d['model']).stem}\n\n"
    md += f"- **Date:** {d['date']}\n"
    md += f"- **Model:** `{d['model']}` (load: {d['model_load_s']:.2f} s)\n"
    md += f"- **Input:** {d['input']} ({d['input_shape'][1]}x{d['input_shape'][0]})\n"
    md += f"- **Pipeline settings:** imgsz={d['imgsz']}, conf={d['conf']}\n"
    md += f"- **Iterations:** {d['iterations']} (+{d['warmup']} warmup)\n\n"

    md += "## Hardware\n\n|  |  |\n|---|---|\n"
    md += _fmt_row(["CPU", f"{info['cpu']} ({info['cpu_cores']}c/{info['cpu_threads']}t)"])
    md += _fmt_row(["RAM", f"{info['ram_gb']} GB"])
    md += _fmt_row(["GPU", f"{info['gpu']} ({info['vram_gb']} GB)" if info["cuda"] else "none (CPU inference)"])
    md += _fmt_row(["Software", f"python {info['python']}, torch {info['torch']}, "
                                f"ultralytics {info['ultralytics']}, opencv {info['opencv']}"])
    md += _fmt_row(["Platform", info["platform"]])

    md += f"\n## Latency (imgsz={d['imgsz']})\n\n"
    md += "| mode | mean ms | median ms | p95 ms | p99 ms | min ms | max ms | std ms | FPS |\n"
    md += "|---|---|---|---|---|---|---|---|---|\n"
    for name, r in [("predict", d["predict"]), ("track (pipeline)", d["track"])]:
        md += _fmt_row([name] + [f"{r[k]:.1f}" for k in
                                 ("mean_ms", "median_ms", "p95_ms", "p99_ms",
                                  "min_ms", "max_ms", "std_ms")] + [f"{r['fps']:.1f}"])
    overhead = d["track"]["mean_ms"] - d["predict"]["mean_ms"]
    md += f"\nTracker (ByteTrack) overhead: **{overhead:.1f} ms/frame**\n"

    md += "\n## Stage breakdown (mean ms, reported by ultralytics)\n\n"
    md += "| mode | preprocess | inference | postprocess |\n|---|---|---|---|\n"
    for name, r in [("predict", d["predict"]), ("track (pipeline)", d["track"])]:
        s = r["stages_mean_ms"]
        md += _fmt_row([name] + [f"{s[k]:.1f}" for k in ("preprocess", "inference", "postprocess")])

    md += "\n## Image size sweep (predict)\n\n"
    md += "| imgsz | mean ms | median ms | p95 ms | FPS |\n|---|---|---|---|---|\n"
    for sz, r in d["imgsz_sweep"].items():
        md += _fmt_row([sz] + [f"{r[k]:.1f}" for k in ("mean_ms", "median_ms", "p95_ms")]
                       + [f"{r['fps']:.1f}"])

    md += "\n## System load during run\n\n"
    md += "| mode | CPU % (all cores) | RSS MB |\n|---|---|---|\n"
    for name, r in [("predict", d["predict"]), ("track (pipeline)", d["track"])]:
        md += _fmt_row([name, f"{r['cpu_percent']:.1f}", f"{r['rss_mb']:.0f}"])
    return md


def run_benchmark(model_path: str, sources, iterations: int = 50) -> Path:
    frame, frame_src = _grab_frame(sources)
    info = _device_info()
    print(f"[bench] {Path(model_path).name} on {info['device']}")
    print(f"[bench] input: {frame_src} ({frame.shape[1]}x{frame.shape[0]})")

    t0 = time.perf_counter()
    model = YOLO(model_path, task="detect")
    load_s = time.perf_counter() - t0

    def predict(f, sz=PIPELINE_IMGSZ):
        return model.predict(f, verbose=False, imgsz=sz, conf=PIPELINE_CONF)

    def track(f):
        return model.track(f, verbose=False, persist=True,
                           imgsz=PIPELINE_IMGSZ, conf=PIPELINE_CONF)

    for _ in range(WARMUP):
        predict(frame)

    print(f"[bench] predict x{iterations} @ imgsz={PIPELINE_IMGSZ}...")
    res_predict = _timed_run(predict, frame, iterations)

    sweep: dict[int, dict] = {}
    n_sweep = max(5, iterations // 5)
    for sz in SWEEP_IMGSZ:
        for _ in range(3):
            predict(frame, sz)
        r = _timed_run(lambda f, sz=sz: predict(f, sz), frame, n_sweep)
        del r["latencies_ms"]
        sweep[sz] = r
        print(f"[bench] imgsz {sz:4d}: {r['mean_ms']:7.1f} ms  ({r['fps']:.1f} fps)")

    # track() must go last: it attaches tracker callbacks to the predictor,
    # which would then also run on every later predict() call and skew it
    print(f"[bench] track   x{iterations} @ imgsz={PIPELINE_IMGSZ}...")
    res_track = _timed_run(track, frame, iterations)

    out_dir = _RUNS_DIR / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "model": model_path,
        "date": datetime.now().isoformat(timespec="seconds"),
        "input": frame_src,
        "input_shape": list(frame.shape),
        "imgsz": PIPELINE_IMGSZ,
        "conf": PIPELINE_CONF,
        "iterations": iterations,
        "warmup": WARMUP,
        "model_load_s": load_s,
        "device": info,
        "predict": res_predict,
        "track": res_track,
        "imgsz_sweep": sweep,
    }
    (out_dir / "benchmark.json").write_text(json.dumps(data, indent=2))
    (out_dir / "benchmark.md").write_text(_report(data))

    print(f"[bench] predict: {res_predict['mean_ms']:.1f} ms/frame "
          f"({res_predict['fps']:.1f} fps)  |  "
          f"track: {res_track['mean_ms']:.1f} ms/frame ({res_track['fps']:.1f} fps)")
    print(f"[bench] results → {out_dir}")
    return out_dir
