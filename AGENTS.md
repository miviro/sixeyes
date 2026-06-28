# SixEyes — Agent Guide

## Project Overview

SixEyes is a real-time computer vision system that tracks objects and drives two servo motors (pan/tilt) via an ESP32 microcontroller. It is a Spanish university TFG (Final Degree Project). The codebase splits into a Python PC application and Arduino C++ ESP32 firmware.

## Repository Layout

```
src/
  sixeyes.py        # Entry point — frame loop, background subtraction
  config.py         # All tunable constants (display, camera, MOG2)
  input_source.py   # Input abstraction (eighteyes / camera / video file / image folder)
  display.py        # Grid renderer for side-by-side panel display
esp32/
  esp32.ino                   # Servo control firmware (pitch GPIO2, yaw GPIO4)
  camera_upload/              # ESP32-CAM firmware — captures MJPEG, serves over WiFi
    camera_upload.ino         # All C++ logic (mDNS, HTTP port 80, MJPEG port 81)
    wifi_config.h             # SSID / password / resolution / JPEG quality
    partitions.csv            # Custom partition table (16 MB flash, 4 MB app)
runs/               # Timestamped output directories (annotated.mp4, errors.csv)
yolo11m.pt          # YOLOv11-medium weights
faces.pt            # Face-detection weights
doguilmak.pt        # Custom-trained weights
Makefile            # flash / flash-cam / start / all targets
pyproject.toml      # Python deps (numpy, opencv-python, pyserial, ultralytics)
shell.nix           # Nix dev environment (includes CUDA, arduino-cli, uv)
```

## Running the Project

```bash
# Flash servo firmware (requires connected ESP32)
make flash PORT=/dev/ttyUSB0

# Flash camera firmware (ESP32-CAM, FQBN esp32:esp32:esp32cam)
make flash-cam CAM_PORT=/dev/ttyUSB0

# Start Python pipeline — auto-discovers the first eighteyes device on the LAN
make start
# or directly:
uv run python src/sixeyes.py yolo11m.pt eighteyes

# Other input sources
uv run python src/sixeyes.py yolo11m.pt camera:0
uv run python src/sixeyes.py yolo11m.pt video path/to/file.mp4
uv run python src/sixeyes.py yolo11m.pt folder path/to/frames/

# Both at once
make all

# 3-D ground-truth mode: add '3d' anywhere in the source list
uv run python src/sixeyes.py yolo11m.pt 3d camera:0 eighteyes
uv run python src/sixeyes.py yolo11m.pt 3d camera:0 eighteyes1 eighteyes2
```

Press `q` in the display window to exit.

## Hardware / Serial Protocol

The ESP32 expects a 3-byte binary command over UART at 115200 baud:

| Byte | Value | Meaning |
|------|-------|---------|
| 0    | `0xFF` | Sync / start-of-frame |
| 1    | 0–180 | Pitch angle (degrees) |
| 2    | 0–180 | Yaw angle (degrees) |

Firmware clamps pitch to 0–140° and yaw to 10–170°. If no command arrives within 500 ms the PWM output is disabled (idle-safe). The Python side sends commands via `pyserial`.

## Key Design Decisions

- **MOG2 background subtraction** (OpenCV `createBackgroundSubtractorMOG2`) is the current active algorithm. Parameters (history, varThreshold, detectShadows) live in `config.py`.
- **YOLO models** (`yolo11m.pt`, `faces.pt`) are present but not wired into the current entry point — they were used in earlier pipeline iterations tracked in `runs/`.
- **Grid display** auto-calculates columns as `ceil(sqrt(n))` so you can add panels by passing more frames to `display.py` without changing layout code.
- **Input generators** in `input_source.py` yield frames lazily; the main loop drives the pace.

## Making Changes

### Adding a new detection algorithm
1. Import or implement it in `src/`.
2. Call it inside the frame loop in `sixeyes.py` after the MOG2 step.
3. Add any new tunables to `config.py` — do not hardcode values in pipeline code.

### Changing servo limits or PWM mapping
Edit the constants at the top of `esp32/esp32.ino` (`PITCH_MIN`, `PITCH_MAX`, `YAW_MIN`, `YAW_MAX`, and the duty-cycle formula). Re-flash with `make flash`.

### Changing camera resolution or WiFi credentials
Edit `esp32/camera_upload/wifi_config.h` (`WIFI_SSID`, `WIFI_PASSWORD`, `CAM_FRAME_SIZE`, `CAM_JPEG_QUALITY`). Resolution is a compile-time constant — re-flash with `make flash-cam`.

### Adding a new input type
Add a generator function in `input_source.py` and extend the `open_input()` routing block.

### Evaluating a run
Each `runs/<timestamp>/` directory contains `annotated.mp4` and `errors.csv` (frame ID, tracking ID, algorithm, angular error in degrees). Use these to compare algorithms.

## Input Source: `eighteyes`

Passing `eighteyes` as the input argument starts `EighteeyesMonitor` — a background daemon that discovers devices via **DNS only** (no TCP connection to port 81, which would displace the live stream since devices only support one viewer at a time).

### Discovery flow

1. `EighteeyesMonitor` sweeps `eighteyes1.local` through `eighteyes16.local` in parallel every 5 s using `concurrent.futures.ThreadPoolExecutor` + `socket.getaddrinfo`.
2. `_sync_monitor()` is called every frame. When a new hostname resolves it calls `_add_source()` once — the source is **never removed** afterwards.
3. `_add_source()` creates a `_FrameBuffer` (background reader thread) and a fresh MOG2 subtractor, then spawns a thread to load the YOLO model asynchronously. Raw + MOG panels appear as soon as the first frame arrives; the YOLO panel follows once the model is ready.
4. `_stream(url)` opens the MJPEG stream via `cv2.VideoCapture` and yields frames. On disconnect it sleeps 1 s and reconnects automatically — the panel stays showing the last received frame in the meantime.

Tunable constant in `config.py`: `EIGHTEYES_MAX_INDEX` (default 16).

No third-party dependencies are needed — `socket` and `concurrent.futures` are stdlib.

## Dependencies

| Tool | Purpose |
|------|---------|
| `uv` | Python package manager / runner |
| `opencv-python` | Video I/O, MOG2, display |
| `numpy` | Array math |
| `pyserial` | Serial communication with ESP32 |
| `ultralytics` | YOLO inference |
| `arduino-cli` | Compile and flash ESP32 firmware |

Install everything via Nix (`nix-shell`) or manually following `README.md`.

## 3-D Ground-Truth Mode (`src/ground_truth.py`)

Pass `3d` as one of the source arguments to enable multi-camera 3-D reconstruction.

```
uv run python src/sixeyes.py yolo11m.pt 3d camera:0 eighteyes
```

### What it does

1. **Intrinsics estimation** — each camera's focal length is derived from known sensor specs:
   - `camera:N` → Creative Live! Cam Sync 1080p, 90° diagonal FOV
   - `eighteyesN` / `:81/` URLs → OV2640 at SVGA, 66° horizontal FOV
   - Defaults to 70° horizontal for unknown sources.
2. **Background pose estimation** — once `GT_BG_FRAMES` quiet frames are collected from ≥ 2 cameras, a background thread extracts ORB features and estimates relative camera poses via the Essential Matrix (`findEssentialMat` + `recoverPose`). The first camera is placed at the world origin.  Scale is not recovered (monocular baseline = 1 world unit).
3. **Object triangulation** — when ≥ 2 cameras both have a YOLO detection and known poses, `cv2.triangulatePoints` is applied for every pair and the results are averaged.
4. **3-D panel** — `get_3d_frame()` renders a perspective wireframe scene (camera frustums, object trail, ground grid, world axes) as a 640×360 BGR image that appears as a panel in the existing grid.

### 3-D view orbit controls (W/S/A/D/=/-; active when `3d` arg is present)

| Key | Action |
|-----|--------|
| `A` / `D` | Yaw left / right |
| `W` / `S` | Pitch up / down |
| `=` / `-` | Zoom in / out |

### Tuning constants in `config.py`

| Constant | Default | Meaning |
|----------|---------|---------|
| `GT_FOV_ESP32CAM` | 66.0 | OV2640 horizontal FOV (degrees) |
| `GT_FOV_CREATIVE_1080P` | 90.0 | Creative 1080p diagonal FOV (degrees) |
| `GT_BG_FRAMES` | 20 | Quiet frames to collect before pose estimation |
| `GT_MIN_MATCHES` | 25 | Minimum ORB inlier matches for Essential Matrix |
| `GT_TRAIL_LEN` | 80 | Object position history length |
| `GT_VIEW_W/H` | 640×360 | 3-D panel size (matches `CELL_W/H`) |
| `GT_VIEW_FOCAL` | 380.0 | Orbit-camera perspective focal length |

### Limitations

- **Scale**: translation between cameras is unit-norm (monocular ambiguity).  The 3-D position is correct in direction but not in absolute metres.
- **Pose requires overlap**: cameras must share some background for ORB matching.  If they face orthogonal directions, pose estimation will fail gracefully (cameras remain at their initialised positions).
- **Single reference**: all cameras are localised relative to the first one.  A full pose graph with loop closure is not implemented.
- **No calibration**: intrinsics are estimated from FOV specs.  Proper calibration (checkerboard) would improve reconstruction accuracy.

## Constraints

- Target latency: ≤33 ms per frame at 30 fps. GPU inference on RTX 3070 Ti runs ~8 ms for a 480×640 frame.
- Camera is configured for 1920×1080 @ 30 fps MJPG. Changing resolution requires updates in both `config.py` and the `input_source.py` `_camera()` function.
- ESP32 PWM duty cycle formula: `duty = 102.375 + angle × 2.16042` (12-bit, 50 Hz). Recalibrate if the servo brand changes.
