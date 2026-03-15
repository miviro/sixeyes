import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

FRAME_W    = 1920
FRAME_H    = 1080
HFOV_DEG   = 69.3
VFOV_DEG   = 42.4
SEQ_LEN    = 20
MIN_SEQ    = 8
PRED_STEPS = 5

INPUT_EMA_ALPHA  = 0.6
OUTPUT_EMA_ALPHA = 0.4

class LSTM(nn.Module):
    def __init__(self, input_size=4, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            batch_first=True, dropout=0.1)
        self.fc   = nn.Linear(hidden, 2)

    def forward(self, x):          # x: (B, T, 4)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])  # (B, 2) — predicts Δ(cx_n, cy_n)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm      = LSTM().to(device)
optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-3)
criterion = nn.MSELoss()

track_history:  dict = defaultdict(list)
track_ema_in:   dict = {}
track_ema_pred: dict = {}

def to_world(cx, cy, w, h, servo_yaw: float, servo_pitch: float) -> list:
    """Convert pixel detection to world-space feature vector.

    World angles are absolute (independent of where the camera is pointing),
    so the LSTM sees true target motion rather than ego-motion artefacts.

    Yaw convention matches aiming.py (yaw is flipped):
        world_yaw = servo_yaw + (0.5 - cx_norm) * HFOV_DEG
    """
    cx_norm = cx / FRAME_W
    cy_norm = cy / FRAME_H
    world_yaw   = servo_yaw   + (0.5 - cx_norm) * HFOV_DEG
    world_pitch = servo_pitch + (cy_norm - 0.5)  * VFOV_DEG
    return [world_yaw, world_pitch, w / FRAME_W, h / FRAME_H]

def train_step(history: list, new_obs: list):
    """Train model to predict delta from last history position to new_obs."""
    lstm.train()
    seq   = torch.tensor([history[-SEQ_LEN:]], dtype=torch.float32, device=device)
    last  = history[-1]
    delta = [new_obs[0] - last[0], new_obs[1] - last[1]]
    y     = torch.tensor([delta], dtype=torch.float32, device=device)
    optimizer.zero_grad()
    criterion(lstm(seq), y).backward()
    optimizer.step()

def predict_future(history: list) -> np.ndarray:
    """Autoregressive PRED_STEPS-ahead prediction via delta accumulation."""
    lstm.eval()
    with torch.no_grad():
        seq     = torch.tensor([history[-SEQ_LEN:]], dtype=torch.float32, device=device)
        pos     = np.array(history[-1][:2], dtype=np.float32)
        for _ in range(PRED_STEPS):
            delta   = lstm(seq)[0].cpu().numpy()        # (2,)
            pos     = pos + delta                       # accumulate displacement
            last_wh = seq[0, -1, 2:].view(1, 1, 2)
            next_xy = torch.tensor(pos, dtype=torch.float32, device=device).view(1, 1, 2)
            next_feat = torch.cat([next_xy, last_wh], dim=2)
            seq = torch.cat([seq[:, 1:], next_feat], dim=1)
    return pos

def update_track(tid: int, cx: float, cy: float, w: float, h: float,
                 servo_yaw: float, servo_pitch: float):
    raw = to_world(cx, cy, w, h, servo_yaw, servo_pitch)

    # EMA smoothing on input
    if tid not in track_ema_in:
        track_ema_in[tid] = raw[:]
    else:
        a = INPUT_EMA_ALPHA
        track_ema_in[tid] = [a * r + (1 - a) * s
                             for r, s in zip(raw, track_ema_in[tid])]
    feat    = track_ema_in[tid]
    history = track_history[tid]

    if len(history) >= MIN_SEQ:
        train_step(history, feat)

    history.append(feat)
    if len(history) > SEQ_LEN + 5:
        del history[0]

    if len(history) < MIN_SEQ:
        return None

    raw_pred = predict_future(history)

    # EMA smoothing on output
    if tid not in track_ema_pred:
        track_ema_pred[tid] = raw_pred.copy()
    else:
        a = OUTPUT_EMA_ALPHA
        track_ema_pred[tid] = a * raw_pred + (1 - a) * track_ema_pred[tid]

    return track_ema_pred[tid]

def calibrating_remaining(tid: int) -> int:
    return max(0, MIN_SEQ - len(track_history[tid]))
