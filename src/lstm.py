import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

FRAME_W    = 640
FRAME_H    = 480
SEQ_LEN    = 20
MIN_SEQ    = 8
PRED_STEPS = 15   # ~0.5 s at 30 fps

class LSTM(nn.Module):
    def __init__(self, input_size=4, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            batch_first=True, dropout=0.1)
        self.fc   = nn.Linear(hidden, 2)

    def forward(self, x):          # x: (B, T, 4)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])  # (B, 2)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm      = LSTM().to(device)
optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-3)
criterion = nn.MSELoss()

track_history: dict = defaultdict(list)

def to_norm(cx, cy, w, h):
    return [cx / FRAME_W, cy / FRAME_H, w / FRAME_W, h / FRAME_H]

def train_step(history: list, new_obs: list):
    lstm.train()
    seq = torch.tensor([history[-SEQ_LEN:]], dtype=torch.float32, device=device)
    y   = torch.tensor([new_obs[:2]],        dtype=torch.float32, device=device)
    optimizer.zero_grad()
    criterion(lstm(seq), y).backward()
    optimizer.step()

def predict_future(history: list) -> np.ndarray:
    lstm.eval()
    with torch.no_grad():
        seq = torch.tensor([history[-SEQ_LEN:]], dtype=torch.float32, device=device)
        pred_xy = None
        for _ in range(PRED_STEPS):
            pred_xy   = lstm(seq)
            last_wh   = seq[0, -1, 2:].view(1, 1, 2)
            next_feat = torch.cat([pred_xy.unsqueeze(1), last_wh], dim=2)
            seq = torch.cat([seq[:, 1:], next_feat], dim=1)
    return pred_xy[0].cpu().numpy()

def update_track(tid: int, cx: float, cy: float, w: float, h: float):
    """Update history for a track, run online training, return prediction or None."""
    feat    = to_norm(cx, cy, w, h)
    history = track_history[tid]

    if len(history) >= MIN_SEQ:
        train_step(history, feat)

    history.append(feat)
    if len(history) > SEQ_LEN + 5:
        del history[0]

    if len(history) >= MIN_SEQ:
        return predict_future(history)
    return None

def calibrating_remaining(tid: int) -> int:
    return max(0, MIN_SEQ - len(track_history[tid]))
