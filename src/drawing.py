import cv2
import numpy as np
from predictors import (
    FRAME_W, FRAME_H, HFOV_DEG, VFOV_DEG, SEQ_LEN,
    PREDICTOR_COLORS, PREDICTOR_LABELS,
)
from aiming import DEADBAND_DEG


def _world_to_px(world_yaw: float, world_pitch: float,
                 servo_yaw: float, servo_pitch: float) -> tuple[int, int]:
    cx_norm = 0.5 - (world_yaw   - servo_yaw)   / HFOV_DEG
    cy_norm = 0.5 + (world_pitch - servo_pitch)  / VFOV_DEG
    px = int(np.clip(cx_norm * FRAME_W, 0, FRAME_W - 1))
    py = int(np.clip(cy_norm * FRAME_H, 0, FRAME_H - 1))
    return px, py


def draw_track(frame, tid, cx, cy, w, h, history,
               servo_yaw: float, servo_pitch: float,
               cal_remaining: int = 0):
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)
    cv2.putText(frame, f"ID {tid}", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if len(history) >= 2:
        pts = [_world_to_px(f[0], f[1], servo_yaw, servo_pitch) for f in history]
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            cv2.line(frame, pts[i - 1], pts[i], (0, int(200 * alpha), 255), 1)

    if cal_remaining > 0:
        cv2.putText(frame, f"cal -{cal_remaining}", (x1, y2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)


def draw_ghost(frame, cx: float, cy: float,
               pred: tuple, algo_id: int, pred_steps: int,
               servo_yaw: float, servo_pitch: float):
    """Draw a ghost marker at the predicted position for one algorithm."""
    px, py = _world_to_px(pred[0], pred[1], servo_yaw, servo_pitch)
    color  = PREDICTOR_COLORS[algo_id]
    label  = PREDICTOR_LABELS[algo_id]
    cv2.arrowedLine(frame, (int(cx), int(cy)), (px, py), color, 1, tipLength=0.2)
    cv2.circle(frame, (px, py), 8, color, 2)
    cv2.circle(frame, (px, py), 2, color, -1)
    cv2.putText(frame, f"{label} t+{pred_steps/30:.1f}s",
                (px + 11, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)


def draw_sweep_hud(frame, patience: int = 0):
    if patience > 0:
        cv2.putText(frame, f"sweep in {patience} steps", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1)
    else:
        cv2.putText(frame, "SWEEP", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
        cv2.putText(frame, "searching...", (8, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)


def draw_hud(frame, sweeping: bool = False, patience: int = 0,
             pred_steps: int = 5, active_ghosts: frozenset = frozenset()):
    if sweeping or patience > 0:
        draw_sweep_hud(frame, patience=patience)
    else:
        cv2.putText(frame, "Trajectory", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"seq={SEQ_LEN}",
                    (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # Ghost status row at bottom
    x = 8
    y = FRAME_H - 10
    for algo_id, label in [(1, "Lin"), (2, "KF"), (3, "LSTM"), (4, "KF+LS")]:
        active = algo_id in active_ghosts
        color  = PREDICTOR_COLORS[algo_id] if active else (80, 80, 80)
        text   = f"[{algo_id}]{label}"
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x += tw + 14

    cv2.putText(frame, f"t+{pred_steps/30:.1f}s ({pred_steps}f) [j/k]",
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Deadband rectangle
    db_half_w = int(DEADBAND_DEG / HFOV_DEG * FRAME_W)
    db_half_h = int(DEADBAND_DEG / VFOV_DEG * FRAME_H)
    cx_c, cy_c = FRAME_W // 2, FRAME_H // 2
    cv2.rectangle(frame,
                  (cx_c - db_half_w, cy_c - db_half_h),
                  (cx_c + db_half_w, cy_c + db_half_h),
                  (0, 180, 255), 1)
    cv2.putText(frame, f"db={DEADBAND_DEG:.1f}deg",
                (cx_c - db_half_w, cy_c - db_half_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 255), 1)
