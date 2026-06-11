import sys
import cv2
from input_source import open_input
from display import make_grid
from config import MOG_DELTA_FRAMES, MOG_VAR_THRESHOLD, MOG_DETECT_SHADOWS

if len(sys.argv) < 2:
    sys.exit("usage: sixeyes.py <camera:N | video | folder>")

mog = cv2.createBackgroundSubtractorMOG2(
    history=MOG_DELTA_FRAMES,
    varThreshold=MOG_VAR_THRESHOLD,
    detectShadows=MOG_DETECT_SHADOWS,
)

for frame in open_input(sys.argv[1]):
    fg_mask = mog.apply(frame)
    fg_bgr  = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

    panels = [
        ("Raw",    frame),
        ("MoG BG", fg_bgr),
    ]
    cv2.imshow("sixeyes", make_grid(panels))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
