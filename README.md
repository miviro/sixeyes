# SixEyes

Prototype pipeline for dual-camera drone tracking with YOLO + ByteTrack and stereo triangulation.

## Quick start

1) Put your YOLO `.pt` file in `model/` and update `config.yaml`.
2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Run:

```bash
python run.py --config config.yaml
```

## Notes

- Inputs are source-agnostic: camera indices (e.g., `0`, `1`), file paths, or stream URLs.
- Tracking uses Ultralytics' ByteTrack integration per camera.
- Stereo triangulation requires a calibration file. Use `calibration/stereo_template.yaml` as a template.
- Without calibration, the program prints `nan` for XYZ.
                                                                                                                                            
  Step 1 — Verify the sign first (before tuning anything)                                                                                     
                                                                                                                                              
  Stand centred in frame. Move your head right. Watch what the servo does:                                                                    
  - If it turns right (chasing you) → sign is correct, proceed to tuning                                                                      
  - If it turns left (running away) → flip the sign: current_yaw += instead of -=

  A wrong sign makes the loop unstable no matter how small KP is.

  ---
  Step 2 — Tune KP alone (KI=KD=0)

  Start extremely small and double upward:
  0.001 → 0.002 → 0.005 → 0.01 → 0.02 → 0.05
  At each step: does it track sluggishly? → go higher. Does it oscillate? → go back one step. The sweet spot is the largest value that doesn't
   oscillate.

  ---
  Step 3 — Add KD to damp overshoot

  Once KP is set, increase KD gradually (~3–10× KP) until overshoot disappears:
  YAW_KD = YAW_KP * 5   # starting point

  ---
  Step 4 — Add KI last, only if there's a steady-state offset

  If the face settles slightly off-centre, add a tiny KI:
  YAW_KI = YAW_KP * 0.1   # starting point

  ---
  Practical note on your current setup

  At KP=0.005 you should be getting at most ~1.5°/frame of correction (for a 300px error). If it's still bouncing at that, the most likely
  cause is detection noise — the bounding box jittering frame-to-frame feeds jittery errors into the P term. That's the scenario where
  bringing the Kalman filter back makes the most sense.
