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
