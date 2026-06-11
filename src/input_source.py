import os
import glob
import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

CAMERA_W = 1920
CAMERA_H = 1080


def open_input(raw_input: str):
    if raw_input.startswith("camera:"):
        return _camera(int(raw_input[len("camera:"):]))
    path = raw_input[len("file:"):] if raw_input.startswith("file:") else raw_input
    if os.path.isdir(path):
        return _folder(path)
    return _video(path)


def _camera(index: int):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
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
