import numpy as np
import torch
import torch.nn as nn
import cv2
from collections import defaultdict, deque

FRAME_W = 1920
FRAME_H = 1080
HFOV_DEG = 69.3
VFOV_DEG = 42.4
SEQ_LEN = 20
MIN_SEQ = 8

INPUT_EMA_ALPHA = 0.6
OUTPUT_EMA_ALPHA = 0.4

PREDICTOR_COLORS = {1: (0, 255, 255), 2: (255, 255, 0), 3: (0, 0, 255), 4: (255, 0, 255)}
PREDICTOR_LABELS = {1: "Lin", 2: "KF", 3: "LSTM", 4: "KF+LS"}

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_world(cx, cy, w, h, servo_yaw: float, servo_pitch: float) -> list:
    cx_norm = cx / FRAME_W
    cy_norm = cy / FRAME_H
    world_yaw   = servo_yaw   + (0.5 - cx_norm) * HFOV_DEG
    world_pitch = servo_pitch + (cy_norm - 0.5)  * VFOV_DEG
    return [world_yaw, world_pitch, w / FRAME_W, h / FRAME_H]


class Predictor:
    name: str = ""
    color: tuple = (128, 128, 128)

    def update(self, tid: int, yaw: float, pitch: float,
               w_norm: float, h_norm: float) -> None:
        raise NotImplementedError

    def predict(self, tid: int, steps: int) -> tuple | None:
        return None

    def calibrating_remaining(self, tid: int) -> int:
        return 0


# ---------------------------------------------------------------------------
# Algorithm 1 — Linear (constant velocity)
# ---------------------------------------------------------------------------

class LinearPredictor(Predictor):
    name = "Lin"
    color = PREDICTOR_COLORS[1]

    def __init__(self, window: int = 5):
        self._obs: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))

    def update(self, tid, yaw, pitch, w_norm, h_norm):
        self._obs[tid].append((yaw, pitch))

    def predict(self, tid, steps):
        obs = self._obs.get(tid)
        if obs is None or len(obs) < 2:
            return None
        pts = list(obs)
        vels = [(pts[i+1][0] - pts[i][0], pts[i+1][1] - pts[i][1])
                for i in range(len(pts) - 1)]
        vy = sum(v[0] for v in vels) / len(vels)
        vp = sum(v[1] for v in vels) / len(vels)
        last = pts[-1]
        return (last[0] + vy * steps, last[1] + vp * steps)

    def calibrating_remaining(self, tid):
        obs = self._obs.get(tid)
        return max(0, 2 - (len(obs) if obs else 0))


# ---------------------------------------------------------------------------
# Algorithm 2 — Kalman filter
# ---------------------------------------------------------------------------

class KalmanPredictor(Predictor):
    name = "KF"
    color = PREDICTOR_COLORS[2]

    def __init__(self):
        self._kfs: dict[int, cv2.KalmanFilter] = {}
        self.prior_pred: dict[int, np.ndarray] = {}  # (yaw, pitch) before measurement update

    def _make_kf(self, yaw: float, pitch: float) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        kf.processNoiseCov    = np.eye(4, dtype=np.float32) * 1e-4
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.array([[yaw], [pitch], [0.0], [0.0]], dtype=np.float32)
        return kf

    def update(self, tid, yaw, pitch, w_norm, h_norm):
        if tid not in self._kfs:
            self._kfs[tid] = self._make_kf(yaw, pitch)
            self.prior_pred[tid] = np.array([yaw, pitch], dtype=np.float32)
            return
        kf = self._kfs[tid]
        prior = kf.predict()
        self.prior_pred[tid] = prior[:2].flatten().astype(np.float32)
        kf.correct(np.array([[yaw], [pitch]], dtype=np.float32))

    def predict(self, tid, steps):
        if tid not in self._kfs:
            return None
        kf = self._kfs[tid]
        F = kf.transitionMatrix.astype(np.float64)
        state = kf.statePost.astype(np.float64)
        Fn = np.linalg.matrix_power(F, steps)
        pred = Fn @ state
        return float(pred[0, 0]), float(pred[1, 0])

    def calibrating_remaining(self, tid):
        return 0 if tid in self._kfs else 1


# ---------------------------------------------------------------------------
# Shared LSTM architecture
# ---------------------------------------------------------------------------

class _LSTMNet(nn.Module):
    def __init__(self, input_size: int = 4, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=0.1)
        self.fc   = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


# ---------------------------------------------------------------------------
# Algorithm 3 — Online-trained LSTM
# ---------------------------------------------------------------------------

class LSTMPredictor(Predictor):
    name = "LSTM"
    color = PREDICTOR_COLORS[3]

    def __init__(self):
        self._criterion = nn.MSELoss()
        self._models:     dict[int, _LSTMNet] = {}
        self._optimizers: dict[int, torch.optim.Optimizer] = {}
        self._history:    dict[int, list] = defaultdict(list)
        self._ema_in:     dict[int, list] = {}
        self._ema_pred:   dict[int, np.ndarray] = {}

    def _get(self, tid: int) -> tuple[_LSTMNet, torch.optim.Optimizer]:
        if tid not in self._models:
            m = _LSTMNet().to(_device)
            self._models[tid]     = m
            self._optimizers[tid] = torch.optim.Adam(m.parameters(), lr=1e-3)
        return self._models[tid], self._optimizers[tid]

    def _train_step(self, tid: int, history: list, new_feat: list):
        model, opt = self._get(tid)
        model.train()
        seq   = torch.tensor([history[-SEQ_LEN:]], dtype=torch.float32, device=_device)
        last  = history[-1]
        delta = [new_feat[0] - last[0], new_feat[1] - last[1]]
        y     = torch.tensor([delta], dtype=torch.float32, device=_device)
        opt.zero_grad()
        self._criterion(model(seq), y).backward()
        opt.step()

    def _predict_future(self, tid: int, history: list, steps: int) -> np.ndarray:
        model, _ = self._get(tid)
        model.eval()
        with torch.no_grad():
            seq = torch.tensor([history[-SEQ_LEN:]], dtype=torch.float32, device=_device)
            pos = np.array(history[-1][:2], dtype=np.float32)
            for _ in range(steps):
                delta   = model(seq)[0].cpu().numpy()
                pos     = pos + delta
                last_wh = seq[0, -1, 2:].view(1, 1, 2)
                next_xy = torch.tensor(pos, dtype=torch.float32, device=_device).view(1, 1, 2)
                seq = torch.cat([seq[:, 1:], torch.cat([next_xy, last_wh], dim=2)], dim=1)
        return pos

    def update(self, tid, yaw, pitch, w_norm, h_norm):
        raw = [yaw, pitch, w_norm, h_norm]
        if tid not in self._ema_in:
            self._ema_in[tid] = raw[:]
        else:
            a = INPUT_EMA_ALPHA
            self._ema_in[tid] = [a * r + (1 - a) * s
                                 for r, s in zip(raw, self._ema_in[tid])]
        feat    = self._ema_in[tid]
        history = self._history[tid]
        if len(history) >= MIN_SEQ:
            self._train_step(tid, history, feat)
        history.append(feat)
        if len(history) > SEQ_LEN + 5:
            del history[0]

    def predict(self, tid, steps):
        history = self._history.get(tid, [])
        if len(history) < MIN_SEQ:
            return None
        raw_pred = self._predict_future(tid, history, steps)
        if tid not in self._ema_pred:
            self._ema_pred[tid] = raw_pred.copy()
        else:
            a = OUTPUT_EMA_ALPHA
            self._ema_pred[tid] = a * raw_pred + (1 - a) * self._ema_pred[tid]
        pred = self._ema_pred[tid]
        return float(pred[0]), float(pred[1])

    def calibrating_remaining(self, tid: int) -> int:
        return max(0, MIN_SEQ - len(self._history.get(tid, [])))


# ---------------------------------------------------------------------------
# Algorithm 4 — Kalman + residual LSTM
# ---------------------------------------------------------------------------

class ResidualLSTMPredictor(Predictor):
    name = "KF+LS"
    color = PREDICTOR_COLORS[4]

    def __init__(self, kalman_ref: KalmanPredictor):
        self._kf        = kalman_ref
        self._criterion = nn.MSELoss()
        self._models:     dict[int, _LSTMNet] = {}
        self._optimizers: dict[int, torch.optim.Optimizer] = {}
        self._err_hist: dict[int, list] = defaultdict(list)
        self._ema_pred: dict[int, np.ndarray] = {}

    def _get(self, tid: int) -> tuple[_LSTMNet, torch.optim.Optimizer]:
        if tid not in self._models:
            m = _LSTMNet(input_size=4).to(_device)
            self._models[tid]     = m
            self._optimizers[tid] = torch.optim.Adam(m.parameters(), lr=1e-3)
        return self._models[tid], self._optimizers[tid]

    def _train_step(self, tid: int, history: list, new_feat: list):
        model, opt = self._get(tid)
        model.train()
        seq   = torch.tensor([history[-SEQ_LEN:]], dtype=torch.float32, device=_device)
        last  = history[-1]
        delta = [new_feat[0] - last[0], new_feat[1] - last[1]]
        y     = torch.tensor([delta], dtype=torch.float32, device=_device)
        opt.zero_grad()
        self._criterion(model(seq), y).backward()
        opt.step()

    def _predict_future(self, tid: int, history: list, steps: int) -> np.ndarray:
        model, _ = self._get(tid)
        model.eval()
        with torch.no_grad():
            seq = torch.tensor([history[-SEQ_LEN:]], dtype=torch.float32, device=_device)
            pos = np.array(history[-1][:2], dtype=np.float32)
            for _ in range(steps):
                delta   = model(seq)[0].cpu().numpy()
                pos     = pos + delta
                last_wh = seq[0, -1, 2:].view(1, 1, 2)
                next_xy = torch.tensor(pos, dtype=torch.float32, device=_device).view(1, 1, 2)
                seq = torch.cat([seq[:, 1:], torch.cat([next_xy, last_wh], dim=2)], dim=1)
        return pos

    def update(self, tid, yaw, pitch, w_norm, h_norm):
        prior = self._kf.prior_pred.get(tid)
        if prior is None:
            return
        yaw_err   = yaw   - float(prior[0])
        pitch_err = pitch - float(prior[1])
        new_feat  = [yaw_err, pitch_err, w_norm, h_norm]
        history   = self._err_hist[tid]
        if len(history) >= MIN_SEQ:
            self._train_step(tid, history, new_feat)
        history.append(new_feat)
        if len(history) > SEQ_LEN + 5:
            del history[0]

    def predict(self, tid, steps):
        history = self._err_hist.get(tid, [])
        if len(history) < MIN_SEQ:
            return None
        kf_pred = self._kf.predict(tid, steps)
        if kf_pred is None:
            return None
        correction = self._predict_future(tid, history, steps)
        if tid not in self._ema_pred:
            self._ema_pred[tid] = correction.copy()
        else:
            a = OUTPUT_EMA_ALPHA
            self._ema_pred[tid] = a * correction + (1 - a) * self._ema_pred[tid]
        c = self._ema_pred[tid]
        return float(kf_pred[0]) + float(c[0]), float(kf_pred[1]) + float(c[1])

    def calibrating_remaining(self, tid: int) -> int:
        kf_rem  = self._kf.calibrating_remaining(tid)
        err_rem = max(0, MIN_SEQ - len(self._err_hist.get(tid, [])))
        return max(kf_rem, err_rem)
