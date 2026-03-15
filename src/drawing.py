import cv2
import numpy as np
from lstm import FRAME_W, FRAME_H, HFOV_DEG, VFOV_DEG, PRED_STEPS, SEQ_LEN, calibrating_remaining
from aiming import DEADBAND_DEG


def _world_to_px(world_yaw: float, world_pitch: float,
                 servo_yaw: float, servo_pitch: float) -> tuple[int, int]:
    """Project a world-space angle back to pixel coordinates."""
    cx_norm = 0.5 - (world_yaw   - servo_yaw)   / HFOV_DEG
    cy_norm = 0.5 + (world_pitch - servo_pitch)  / VFOV_DEG
    px = int(np.clip(cx_norm * FRAME_W, 0, FRAME_W - 1))
    py = int(np.clip(cy_norm * FRAME_H, 0, FRAME_H - 1))
    return px, py


def draw_track(frame, tid, cx, cy, w, h, history, pred, servo_yaw: float, servo_pitch: float):
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"ID {tid}", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if len(history) >= 2:
        pts = [_world_to_px(f[0], f[1], servo_yaw, servo_pitch) for f in history]
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            cv2.line(frame, pts[i - 1], pts[i], (0, int(200 * alpha), 255), 1)

    if pred is not None:
        px, py = _world_to_px(pred[0], pred[1], servo_yaw, servo_pitch)
        cv2.arrowedLine(frame, (int(cx), int(cy)), (px, py), (0, 0, 255), 2, tipLength=0.25)
        cv2.circle(frame, (px, py), 7, (0, 0, 255), -1)
        cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)
        cv2.putText(frame, f"t+{PRED_STEPS / 30:.1f}s", (px + 9, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    else:
        cv2.putText(frame, f"cal -{calibrating_remaining(tid)}", (x1, y2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)


def draw_hud(frame):
    cv2.putText(frame, "Trajectory", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"seq={SEQ_LEN}  pred=+{PRED_STEPS}f ({PRED_STEPS/30:.1f}s)",
                (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # Deadband rectangle — region where servo won't move
    db_half_w = int(DEADBAND_DEG / HFOV_DEG * FRAME_W)
    db_half_h = int(DEADBAND_DEG / VFOV_DEG * FRAME_H)
    cx, cy = FRAME_W // 2, FRAME_H // 2
    cv2.rectangle(frame,
                  (cx - db_half_w, cy - db_half_h),
                  (cx + db_half_w, cy + db_half_h),
                  (0, 180, 255), 1)
    cv2.putText(frame, f"db={DEADBAND_DEG:.1f}deg",
                (cx - db_half_w, cy - db_half_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 255), 1)
