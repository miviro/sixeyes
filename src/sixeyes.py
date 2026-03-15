import cv2
from ultralytics import YOLO
from lstm import FRAME_W, FRAME_H, update_track, track_history
from drawing import draw_track, draw_hud

yolo = YOLO("faces.pt")
cap  = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, 30)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    results = yolo.track(frame, persist=True, verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        tids  = results[0].boxes.id.cpu().numpy().astype(int)

        for (cx, cy, w, h), tid in zip(boxes, tids):
            pred = update_track(tid, cx, cy, w, h)
            draw_track(frame, tid, cx, cy, w, h, track_history[tid], pred)

    draw_hud(frame)
    cv2.imshow("sixeyes", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
