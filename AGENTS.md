# SixEyes — Agent Guide

## What this project is

SixEyes is a real-time object-tracking system built as a Spanish university TFG. A PC running Python detects objects with YOLO and MOG2 background subtraction across one or more video sources, then drives a two-axis pan/tilt servo mount via an ESP32 over serial.

Two separate ESP32 firmwares exist:

| Sketch | Board | Role |
|--------|-------|------|
| `esp32/pantilt/` | ESP32 (generic) | Receives 3-byte serial commands from Python and drives two servos |
| `esp32/eighteyes/` | AI Thinker ESP32-CAM | Connects to WiFi, captures MJPEG, streams it over the LAN |

---

## Repository layout

```
src/
  sixeyes.py        # Entry point — arg parsing, frame loop, display
  aiming.py         # Servo state machine: deadband, rate-limit, sweep, serial I/O
  config.py         # All tuneable constants
  display.py        # make_grid() — renders panels into a tiled OpenCV window
  input_source.py   # Input generators: camera, video, folder, MJPEG URL, eighteyes
esp32/
  pantilt/
    pantilt.ino              # Servo firmware (yaw GPIO2, pitch GPIO4)
    manual_servo.ino.ignore  # Manual test sketch (not compiled)
  eighteyes/
    eighteyes.ino            # Camera firmware (WiFi + MJPEG server)
    wifi_config.h            # SSID / password / resolution / JPEG quality
    partitions.csv           # Custom 16 MB partition table
Makefile            # flash / flash-cam / start / all targets
pyproject.toml      # Python deps (uv-managed)
shell.nix           # Nix dev shell (arduino-cli, uv, CUDA)
runs/               # Timestamped output directories from past sessions
```

---

## Running the project

```bash
# Flash pan/tilt firmware
make flash PORT=/dev/ttyUSB0

# Flash ESP32-CAM firmware
make flash-cam CAM_PORT=/dev/ttyUSB0

# Start Python pipeline (auto-discovers eighteyes cameras on the LAN)
make start
# equivalent:
uv run python src/sixeyes.py yolo11m.pt eighteyes

# With servo tracking enabled
uv run python src/sixeyes.py yolo11m.pt eighteyes --follow
uv run python src/sixeyes.py yolo11m.pt eighteyes --follow --port /dev/ttyUSB1

# Other input sources
uv run python src/sixeyes.py yolo11m.pt camera:0
uv run python src/sixeyes.py yolo11m.pt path/to/file.mp4
uv run python src/sixeyes.py yolo11m.pt path/to/frames/

# Multiple sources at once
uv run python src/sixeyes.py yolo11m.pt camera:0 eighteyes
```

Press `q` to quit.

---

## CLI flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--follow` | off | Enable servo tracking via serial |
| `--port DEV` | `/dev/ttyUSB0` | Serial port used by `--follow` |

Positional arguments after the model path are input sources.

---

## Python pipeline (`src/sixeyes.py`)

The main loop runs once per display frame. For every registered source:

1. `_FrameBuffer.get()` — grab the latest frame from the background reader thread.
2. MOG2 background subtraction — run on a downscaled copy (`MOG_SCALE_W × MOG_SCALE_H`) then upscaled back, yielding a foreground mask panel.
3. YOLO tracking — `model.track(frame, persist=True, imgsz=128, conf=0.1)` on the full frame. The model is loaded asynchronously in a background thread when a source is first registered; the YOLO panel is absent until it is ready.
4. Dead-zone check — a green rectangle covering the centre `FOLLOW_ZONE` fraction of the frame is drawn on the YOLO panel. When `--follow` is active and the highest-confidence detection is **outside** the rectangle, `aiming.aim()` is called.
5. `aiming.sweep_tick()` — called after `SWEEP_PATIENCE` (30) consecutive frames with no detection, sweeping the servos in a triangle/sine pattern.
6. `make_grid()` renders all panels into a tiled window. A status bar at the bottom shows EMA frame time, FPS, active sources, and follow state.

---

## Servo aiming (`src/aiming.py`)

Stateful module; call `init_serial()` once, then call `update_estimated_angles(dt)` and optionally `aim()` every frame.

### Key constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `YAW_CENTER` | 90° | Neutral yaw (straight ahead) |
| `PITCH_CENTER` | 70° | Neutral pitch (midpoint of 0–140°) |
| `YAW_MIN / YAW_MAX` | 10° / 170° | Hardware limits |
| `PITCH_MIN / PITCH_MAX` | 0° / 140° | Hardware limits |
| `SERVO_SPEED_DEG_S` | 300 °/s | Used to estimate current servo position |
| `SEND_INTERVAL` | 0.5 s | Minimum time between serial writes |
| `DEADBAND_DEG` | 7.5° | Skip `aim()` if target is already close enough |
| `SWEEP_YAW_PERIOD` | 10 s | Full yaw sweep cycle |
| `SWEEP_PITCH_AMP` | 20° | Pitch amplitude during sweep |
| `SWEEP_PITCH_CYCLES` | 5 | Pitch oscillations per yaw cycle |

### Pixel → world-space conversion (done in `sixeyes.py`)

```python
world_yaw   = servo_yaw   + (0.5 - tx / fw) * HFOV_DEG
world_pitch = servo_pitch + (ty / fh - 0.5) * VFOV_DEG
```

`HFOV_DEG = 69.3°`, `VFOV_DEG = 42.4°` (Creative Live! Cam Sync 1080p). These live in `config.py`.

### Serial write format

`_send_raw()` and `aim()` both write exactly 3 bytes and immediately flush:

```
0xFF  |  pitch (0–140)  |  yaw (10–170)
```

`0xFF` is unambiguous as sync because valid angle values never reach 255.

---

## Pan/tilt firmware (`esp32/pantilt/pantilt.ino`)

| Property | Value |
|----------|-------|
| Board | Generic ESP32 (FQBN `esp32:esp32:esp32`) |
| Yaw servo | GPIO 2 |
| Pitch servo | GPIO 4 |
| PWM freq | 50 Hz, 12-bit resolution |
| Baud | 115200 |
| Idle timeout | 500 ms — PWM duty set to 0 after silence |

**Duty formula:** `duty = 102.375 + angle × 2.16042` (12-bit, 50 Hz). `ledcWrite` receives `4095 - duty` (inverted).

**Parser state machine:** `WAIT_SYNC → READ_PITCH → READ_YAW → WAIT_SYNC`. The byte `0xFF` always resets to `READ_PITCH`. After a complete frame, both servos are written and the idle timer resets. If no frame arrives within 500 ms, `ledcWrite(pin, 0)` is called (servo goes limp).

---

## Eighteyes camera firmware (`esp32/eighteyes/`)

| Property | Value |
|----------|-------|
| Board | AI Thinker ESP32-CAM (FQBN `esp32:esp32:esp32cam`) |
| Chip | ESP32-D0WD-V3, 16 MB flash, PSRAM required |
| Sensor | OV2640 |
| LED | GPIO 33, active-LOW |

**Boot sequence:**

1. 5 rapid LED blinks → `setup()` started.
2. Connects to WiFi as station. Scans mDNS for existing `eighteyesN.local` names and claims the lowest unused N (1–16). Registers as `eighteyesN.local`.
3. 1 blink → WiFi + mDNS done. 2 blinks → camera initialised. 10 rapid blinks → servers started.
4. `ledFatal(N)` (N long + 8 rapid blinks, loops forever) on failure at stage N.

**Servers:**

| Port | Purpose |
|------|---------|
| 80 | HTML viewer (`<img>` streaming from port 81) |
| 81 | MJPEG stream (`multipart/x-mixed-replace; boundary=frame`) |

Only one stream client is served at a time; port 80 is unresponsive while streaming (harmless — the browser already loaded the page).

**Configuration** — edit `esp32/eighteyes/wifi_config.h` before flashing:

| Define | Current | Meaning |
|--------|---------|---------|
| `WIFI_SSID` | set | Network name |
| `WIFI_PASSWORD` | set | Network password |
| `CAM_FRAME_SIZE` | `FRAMESIZE_VGA` (640×480) | Capture resolution |
| `CAM_JPEG_QUALITY` | `10` | 0–63, lower = better quality |

Resolution is a compile-time constant — changing it requires reflashing.

---

## Input source system (`src/input_source.py`)

`open_input(spec)` returns a generator that yields BGR frames.

| `spec` prefix | Behaviour |
|---------------|-----------|
| `camera:N` | Opens `/dev/videoN` at 1920×1080 MJPG 30 fps |
| `http://` / `https://` | MJPEG stream — reconnects automatically on disconnect |
| directory path | Yields images in sorted order (jpg/png/bmp/tiff) |
| file path | Reads video with `cv2.VideoCapture` |

`EighteeyesMonitor` probes `eighteyes1..16.local` via DNS every 5 s using `concurrent.futures.ThreadPoolExecutor`. On resolution, it returns `http://<ip>:81/` and the URL is handed to `_stream()`. Devices are never removed once found.

---

## Display system (`src/display.py`)

`make_grid(panels)` takes a list of `(title, frame)` pairs and renders them into a dark-background tiled grid. Each cell is `640×360` px (`CELL_W × CELL_H`). Column count = `ceil(sqrt(n))`. Frames are resized with `cv2.resize`; a title bar of 28 px sits above each cell.

---

## Config reference (`src/config.py`)

### Display
| Constant | Value |
|----------|-------|
| `CELL_W / CELL_H` | 640 / 360 |
| `TITLE_H` | 28 px |
| `GAP` | 4 px |

### Camera
| Constant | Value |
|----------|-------|
| `CAMERA_W / CAMERA_H` | 1920 / 1080 |
| `CAMERA_FPS` | 30 |
| `HFOV_DEG` | 69.3° |
| `VFOV_DEG` | 42.4° |

### Follow / tracking
| Constant | Value |
|----------|-------|
| `FOLLOW_ZONE` | 0.5 (centre 50% of each axis is the dead zone) |
| `FOLLOW_PORT` | `/dev/ttyUSB0` |

### MOG2
| Constant | Value |
|----------|-------|
| `MOG_DELTA_FRAMES` | 200 |
| `MOG_VAR_THRESHOLD` | 16 |
| `MOG_DETECT_SHADOWS` | True |
| `MOG_SCALE_W / H` | 192 / 108 |

---

## Making changes

### Changing servo GPIO or PWM mapping
Edit constants at the top of `esp32/pantilt/pantilt.ino` (`yawPin`, `pitchPin`, `DUTY_OFFSET`, `DUTY_SCALE`). Reflash with `make flash`.

### Swapping yaw/pitch direction
Negate the `dx` or `dy` multiplier in `sixeyes.py` around line 200 (the `world_yaw` / `world_pitch` computation).

### Changing camera resolution or credentials
Edit `esp32/eighteyes/wifi_config.h` and reflash with `make flash-cam`. Resolution is compile-time only.

### Adding a detection algorithm
Add it in `src/` and call it inside the frame loop in `sixeyes.py` after the MOG2 step. Add tunables to `config.py`.

### Adding a new input type
Add a generator function in `input_source.py` and extend the routing block in `open_input()`.

### Changing the dead zone size
Edit `FOLLOW_ZONE` in `config.py` (fraction of frame, centred; default 0.5 = centre half).

### Changing servo speed estimate or deadband
Edit `SERVO_SPEED_DEG_S`, `DEADBAND_DEG`, or `SEND_INTERVAL` at the top of `aiming.py`.

---

## Dependencies

| Tool | Purpose |
|------|---------|
| `uv` | Python package manager / runner |
| `opencv-python` | Video I/O, MOG2, display |
| `numpy` | Array math |
| `pyserial` | Serial communication with pan/tilt ESP32 |
| `ultralytics` | YOLO inference |
| `arduino-cli` | Compile and flash ESP32 firmware |

Install everything via Nix (`nix-shell`) or manually — `arduino-cli` needs the `esp32:esp32` core.
