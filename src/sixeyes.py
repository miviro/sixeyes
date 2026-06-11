import sys
import cv2
from input_source import open_input

if len(sys.argv) < 2:
    sys.exit("usage: sixeyes.py <camera:N | video | folder>")

for frame in open_input(sys.argv[1]):
    cv2.imshow("sixeyes", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
