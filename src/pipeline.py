from predictors import (
    to_world, Predictor,
    LinearPredictor, KalmanPredictor, LSTMPredictor, ResidualLSTMPredictor,
)


class TrackingPipeline:
    def __init__(self):
        kf = KalmanPredictor()
        self.predictors: dict[int, Predictor] = {
            1: LinearPredictor(),
            2: kf,
            3: LSTMPredictor(),
            4: ResidualLSTMPredictor(kalman_ref=kf),
        }

    def step(self,
             tid: int,
             cx: float, cy: float, w: float, h: float,
             servo_yaw: float, servo_pitch: float,
             pred_steps: int,
             ) -> dict[int, tuple | None]:
        """Run predictor pass (c) + differences (d) for one tracked object.

        Returns {algo_id: (world_yaw, world_pitch) | None} for drawing.
        The differences step is handled inside ResidualLSTMPredictor.update(),
        which reads KalmanPredictor.prior_pred set during the KF update.
        Predictor insertion order (1→2→3→4) ensures KF runs before residual LSTM.
        """
        feat = to_world(cx, cy, w, h, servo_yaw, servo_pitch)
        yaw, pitch, w_norm, h_norm = feat

        for p in self.predictors.values():
            p.update(tid, yaw, pitch, w_norm, h_norm)

        return {algo_id: p.predict(tid, pred_steps)
                for algo_id, p in self.predictors.items()}

    def history(self, tid: int) -> list:
        """World-space observation history from the LSTM predictor (for trail drawing)."""
        return self.predictors[3]._history.get(tid, [])

    def calibrating_remaining(self, tid: int) -> int:
        return self.predictors[3].calibrating_remaining(tid)
