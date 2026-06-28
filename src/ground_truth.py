"""3-D ground-truth reconstruction from multiple camera views.

Pipeline
--------
1. Each camera calls update() with its latest frame, MOG foreground mask, and
   (optionally) a YOLO detection centre in pixel coordinates.
2. Quiet background frames (foreground ratio < 8 %) are stored.  Once
   GT_BG_FRAMES frames have been collected for ≥ 2 cameras, a background
   thread runs ORB feature extraction + Essential-Matrix pose estimation.
3. Once ≥ 2 cameras have known poses and simultaneous detections, the object
   is triangulated via cv2.triangulatePoints; multiple pairs are averaged.
4. get_3d_frame() renders the 3-D scene (camera frustums, object trail) to a
   BGR OpenCV image that plugs straight into the make_grid() panel system.

Keyboard controls (call handle_key() with the cv2.waitKey() result):
    W/S  — orbit pitch   A/D — orbit yaw   =/- — zoom in/out
"""

import threading
import numpy as np
import cv2
from config import (
    GT_FOV_ESP32CAM, GT_FOV_CREATIVE_1080P, GT_FOV_DEFAULT,
    GT_BG_FRAMES, GT_MIN_MATCHES, GT_TRAIL_LEN,
    GT_VIEW_W, GT_VIEW_H, GT_VIEW_FOCAL,
)

# Resolution used for ORB background-feature extraction (small = faster matching)
_MATCH_W, _MATCH_H = 320, 240

_CAM_COLORS = [
    (255, 140,   0),   # orange
    (  0, 200, 255),   # cyan
    (255,   0, 180),   # magenta
    (160, 255,   0),   # yellow-green
    (  0, 255, 140),   # spring-green
    (200,   0, 255),   # purple
]


# ---------------------------------------------------------------------------
# Intrinsics helpers
# ---------------------------------------------------------------------------

def _intrinsics(key: str, shape: tuple) -> np.ndarray:
    """Return 3×3 camera matrix K estimated from source type and frame size."""
    h, w = shape[:2]
    if key.startswith("eighteyes") or ":81/" in key:
        # OV2640 at SVGA 800×600, horizontal FOV
        f = (w / 2.0) / np.tan(np.radians(GT_FOV_ESP32CAM / 2.0))
    elif key.startswith("camera:"):
        # Creative Live! Cam Sync 1080p, diagonal FOV
        diag = np.hypot(w, h)
        f = (diag / 2.0) / np.tan(np.radians(GT_FOV_CREATIVE_1080P / 2.0))
    else:
        f = (w / 2.0) / np.tan(np.radians(GT_FOV_DEFAULT / 2.0))
    return np.array([[f, 0.0, w / 2.0],
                     [0.0, f, h / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _scale_K(K: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Scale intrinsic matrix K to a different image resolution."""
    sx = new_w / (K[0, 2] * 2.0)
    sy = new_h / (K[1, 2] * 2.0)
    return np.diag([sx, sy, 1.0]) @ K


# ---------------------------------------------------------------------------
# 3-D wireframe renderer
# ---------------------------------------------------------------------------

class _View3D:
    """Orbit-camera perspective renderer to an OpenCV BGR image."""

    def __init__(self, w: int, h: int, focal: float) -> None:
        self.w, self.h = w, h
        self.focal = focal
        self.yaw: float = 0.4     # orbit yaw  (radians)
        self.pitch: float = 0.35  # orbit pitch (radians, positive = look down)
        self.dist: float = 5.0    # distance from scene origin

    # ---- internal projection ----

    def _project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project (N,3) world points → (N,2) pixel coords + (N,) valid mask."""
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        cam_pos = self.dist * np.array([sy * cp, sp, cy * cp])
        fwd = -cam_pos / (np.linalg.norm(cam_pos) + 1e-9)
        wu = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, wu)
        rn = np.linalg.norm(right)
        right = right / rn if rn > 1e-6 else np.array([1.0, 0.0, 0.0])
        up = np.cross(right, fwd)
        R_v = np.array([right, up, -fwd])       # rows → view axes
        pv = (pts - cam_pos) @ R_v.T            # (N, 3) in view space
        valid = pv[:, 2] > 0.01
        px = np.full((len(pts), 2), -1.0)
        if valid.any():
            z = pv[valid, 2]
            px[valid, 0] = self.focal * pv[valid, 0] / z + self.w / 2.0
            px[valid, 1] = -self.focal * pv[valid, 1] / z + self.h / 2.0
        return px, valid

    @staticmethod
    def _ipt(px: np.ndarray, i: int) -> tuple[int, int]:
        return (int(px[i, 0]), int(px[i, 1]))

    # ---- public render ----

    def render(self, cameras: list, obj_pos, trail: list) -> np.ndarray:
        """
        cameras : list of (center_3d, R_3x3, color_bgr, label)
                  R is world-to-camera rotation (OpenCV convention).
        obj_pos : (3,) ndarray or None
        trail   : list of (3,) ndarrays (oldest first)
        """
        img = np.full((self.h, self.w, 3), 25, dtype=np.uint8)

        # ground grid (XZ plane, Y = 0)
        sg, eg = [], []
        for i in range(-5, 6):
            sg += [[i, 0, -5], [-5, 0, i]]
            eg += [[i, 0,  5], [ 5, 0, i]]
        apts = np.array(sg + eg, dtype=np.float64)
        pxg, vg = self._project(apts)
        n = len(sg)
        for i in range(n):
            if vg[i] and vg[i + n]:
                cv2.line(img, self._ipt(pxg, i), self._ipt(pxg, i + n), (42, 42, 42), 1)

        # world axes at origin
        ax = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float64)
        pxa, va = self._project(ax)
        for i, col in enumerate([(0, 0, 160), (0, 140, 0), (140, 0, 0)], 1):
            if va[0] and va[i]:
                cv2.arrowedLine(img, self._ipt(pxa, 0), self._ipt(pxa, i),
                                col, 2, tipLength=0.4)

        # camera frustums
        for center, R, color, label in cameras:
            # R is world→camera; rows are camera-space axes expressed in world coords.
            # R[2] = forward direction of camera in world space.
            fwd = R[2]; rgt = R[0]; dwn = R[1]
            sz, dep = 0.25, 0.9
            base = center + fwd * dep
            corn = np.array([
                base + rgt * sz - dwn * sz,
                base - rgt * sz - dwn * sz,
                base - rgt * sz + dwn * sz,
                base + rgt * sz + dwn * sz,
            ])
            all_c = np.vstack([[center], corn])
            pxc, vc = self._project(all_c)
            if vc[0]:
                for i in range(1, 5):
                    if vc[i]:
                        cv2.line(img, self._ipt(pxc, 0), self._ipt(pxc, i), color, 2)
                for i in range(1, 5):
                    j = 1 + i % 4
                    if vc[i] and vc[j]:
                        cv2.line(img, self._ipt(pxc, i), self._ipt(pxc, j), color, 1)
                pt0 = self._ipt(pxc, 0)
                cv2.putText(img, label, (pt0[0] + 4, pt0[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        # object trail
        if len(trail) > 1:
            tpts = np.array(trail, dtype=np.float64)
            pxt, vt = self._project(tpts)
            for i in range(len(trail) - 1):
                if vt[i] and vt[i + 1]:
                    a = (i + 1) / len(trail)
                    cv2.line(img, self._ipt(pxt, i), self._ipt(pxt, i + 1),
                             (int(20 * a), int(210 * a), int(70 * a)), 1)

        # tracked object
        if obj_pos is not None:
            pxo, vo = self._project(np.array([obj_pos]))
            if vo[0]:
                cv2.circle(img, self._ipt(pxo, 0), 10, (0, 255, 70), -1)
                cv2.circle(img, self._ipt(pxo, 0), 10, (255, 255, 255), 1)

        # HUD
        cv2.putText(img, "W/S: pitch  A/D: yaw  =/-: zoom",
                    (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)
        info = (f"yaw={np.degrees(self.yaw):.0f}°  "
                f"pitch={np.degrees(self.pitch):.0f}°  "
                f"dist={self.dist:.1f}")
        cv2.putText(img, info, (5, self.h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)

        n_cams = len(cameras)
        obj_str = (f"obj=({obj_pos[0]:.2f},{obj_pos[1]:.2f},{obj_pos[2]:.2f})"
                   if obj_pos is not None else "obj=none")
        cv2.putText(img, f"{n_cams} cam(s)  {obj_str}",
                    (5, self.h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)
        return img


# ---------------------------------------------------------------------------
# Ground-truth engine
# ---------------------------------------------------------------------------

class GroundTruthEngine:
    """Multi-camera 3-D reconstruction.  All public methods are thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cameras: dict[str, dict] = {}
        self._order: list[str] = []               # insertion order of camera keys
        self._detections: dict[str, tuple | None] = {}
        self._obj_3d: np.ndarray | None = None
        self._trail: list[np.ndarray] = []
        self._view = _View3D(GT_VIEW_W, GT_VIEW_H, GT_VIEW_FOCAL)
        self._pose_thread: threading.Thread | None = None

    # ---- public API ----

    def add_camera(self, key: str, spec: str) -> None:
        """Register a new camera.  Safe to call before or after first frame."""
        with self._lock:
            if key in self._cameras:
                return
            self._cameras[key] = {
                'spec': spec,
                'K':    None,           # full-resolution intrinsic matrix
                'K_sm': None,           # intrinsics scaled to _MATCH_W × _MATCH_H
                'R': np.eye(3),         # world-to-camera rotation
                't': np.zeros((3, 1)),  # world-to-camera translation
                'P': None,              # 3×4 projection matrix (full resolution)
                'pose_known': False,
                'bg_frames': [],        # grayscale _MATCH_W × _MATCH_H background frames
                'kp': None,             # ORB keypoints for background
                'des': None,            # ORB descriptors for background
                'center_3d': np.zeros(3),
                'color': _CAM_COLORS[len(self._order) % len(_CAM_COLORS)],
            }
            self._order.append(key)
            self._detections[key] = None

    def update(self, key: str, frame: np.ndarray,
               fg_mask: np.ndarray, det_px: tuple | None) -> None:
        """
        Call once per frame for each camera.

        key     : camera identifier (same as passed to add_camera)
        frame   : full-resolution BGR frame
        fg_mask : single-channel MOG foreground mask, same spatial size as frame
        det_px  : (cx, cy) in full-resolution pixels, or None
        """
        with self._lock:
            cam = self._cameras.get(key)
            if cam is None:
                return

            # Initialise intrinsics from the first frame of each camera
            if cam['K'] is None:
                cam['K']    = _intrinsics(key, frame.shape)
                cam['K_sm'] = _scale_K(cam['K'], _MATCH_W, _MATCH_H)
                # First camera becomes the world-frame origin
                if not any(c['pose_known'] for c in self._cameras.values()):
                    cam['pose_known'] = True
                    cam['R'] = np.eye(3)
                    cam['t'] = np.zeros((3, 1))
                    cam['P'] = cam['K'] @ np.hstack([np.eye(3), np.zeros((3, 1))])
                    cam['center_3d'] = np.zeros(3)

            # Collect quiet background frames for pose estimation
            if len(cam['bg_frames']) < GT_BG_FRAMES:
                fg_ratio = np.count_nonzero(fg_mask) / max(fg_mask.size, 1)
                if fg_ratio < 0.08:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cam['bg_frames'].append(cv2.resize(gray, (_MATCH_W, _MATCH_H)))
                    if len(cam['bg_frames']) == GT_BG_FRAMES:
                        self._maybe_start_pose_thread()

            self._detections[key] = det_px
            if det_px is not None:
                self._triangulate()

    def get_3d_frame(self) -> np.ndarray:
        """Render the current 3-D view as a BGR image (GT_VIEW_W × GT_VIEW_H)."""
        with self._lock:
            cams = [
                (self._cameras[k]['center_3d'],
                 self._cameras[k]['R'],
                 self._cameras[k]['color'],
                 k)
                for k in self._order
                if self._cameras[k]['pose_known']
            ]
            return self._view.render(cams, self._obj_3d, list(self._trail))

    def handle_key(self, key: int) -> None:
        """Update orbit-camera parameters from a cv2.waitKey() result."""
        v = self._view
        if   key == ord('a'): v.yaw -= 0.08
        elif key == ord('d'): v.yaw += 0.08
        elif key == ord('w'): v.pitch = min(1.5, v.pitch + 0.08)
        elif key == ord('s'): v.pitch = max(-1.5, v.pitch - 0.08)
        elif key == ord('='): v.dist  = max(0.5, v.dist - 0.3)
        elif key == ord('-'): v.dist += 0.3

    # ---- background pose estimation ----

    def _maybe_start_pose_thread(self) -> None:
        if self._pose_thread is not None and self._pose_thread.is_alive():
            return
        self._pose_thread = threading.Thread(
            target=self._run_pose_estimation, daemon=True, name="gt-pose")
        self._pose_thread.start()

    def _run_pose_estimation(self) -> None:
        # Snapshot camera data under lock (copy what we need for off-lock computation)
        with self._lock:
            snapshot: dict[str, dict] = {}
            for k in self._order:
                c = self._cameras[k]
                if len(c['bg_frames']) >= GT_BG_FRAMES and c['K_sm'] is not None:
                    snapshot[k] = {
                        'bg_frame':   c['bg_frames'][len(c['bg_frames']) // 2],
                        'K_sm':       c['K_sm'].copy(),
                        'K':          c['K'].copy(),
                        'pose_known': c['pose_known'],
                        'R':          c['R'].copy(),
                        't':          c['t'].copy(),
                        'kp':         c['kp'],
                        'des':        c['des'],
                    }
        if not snapshot:
            return

        orb = cv2.ORB_create(nfeatures=600)
        bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Extract ORB features for every snapshotted camera
        for k, s in snapshot.items():
            if s['des'] is None:
                kp, des = orb.detectAndCompute(s['bg_frame'], None)
                s['kp'], s['des'] = kp, des
                with self._lock:
                    self._cameras[k]['kp']  = kp
                    self._cameras[k]['des'] = des

        # Find reference: first camera with known pose + features
        ref_key = None
        for k in self._order:
            s = snapshot.get(k)
            if s and s['pose_known'] and s['des'] is not None:
                ref_key = k
                break
        if ref_key is None:
            return

        ref = snapshot[ref_key]

        for k, s in snapshot.items():
            if k == ref_key or s['pose_known'] or s['des'] is None:
                continue

            try:
                matches = bf.match(ref['des'], s['des'])
            except Exception:
                continue
            matches = sorted(matches, key=lambda m: m.distance)[:300]

            if len(matches) < GT_MIN_MATCHES:
                print(f"[3D] {k}: {len(matches)} matches < {GT_MIN_MATCHES}, skipping")
                continue

            pts1 = np.float32([ref['kp'][m.queryIdx].pt for m in matches])
            pts2 = np.float32([  s['kp'][m.trainIdx].pt for m in matches])

            E, mask_e = cv2.findEssentialMat(
                pts1, pts2, ref['K_sm'],
                method=cv2.RANSAC, prob=0.999, threshold=1.0,
            )
            if E is None:
                continue
            inl1 = pts1[mask_e.ravel() == 1]
            inl2 = pts2[mask_e.ravel() == 1]
            if len(inl1) < 8:
                continue

            _, R_rel, t_rel, _ = cv2.recoverPose(E, inl1, inl2, ref['K_sm'])

            # Compose relative pose with the reference camera's world pose
            R_new = R_rel @ ref['R']
            t_new = R_rel @ ref['t'] + t_rel
            center_new = (-R_new.T @ t_new).ravel()

            with self._lock:
                c = self._cameras[k]
                if not c['pose_known']:
                    c['R']          = R_new
                    c['t']          = t_new
                    c['center_3d']  = center_new
                    c['P']          = c['K'] @ np.hstack([R_new, t_new])
                    c['pose_known'] = True
                    print(f"[3D] Pose set for {k}: "
                          f"center=({center_new[0]:.3f}, "
                          f"{center_new[1]:.3f}, "
                          f"{center_new[2]:.3f})")

    # ---- triangulation (called with self._lock held) ----

    def _triangulate(self) -> None:
        avail = []
        for k in self._order:
            c = self._cameras[k]
            d = self._detections.get(k)
            if c['pose_known'] and c['P'] is not None and d is not None:
                avail.append((c['P'], d))

        if len(avail) < 2:
            return

        positions = []
        for i in range(len(avail)):
            for j in range(i + 1, len(avail)):
                P1, d1 = avail[i]
                P2, d2 = avail[j]
                pt1 = np.float32([[d1[0]], [d1[1]]])
                pt2 = np.float32([[d2[0]], [d2[1]]])
                X = cv2.triangulatePoints(P1, P2, pt1, pt2)
                w = X[3, 0]
                if abs(w) > 1e-9:
                    positions.append(X[:3, 0] / w)

        if positions:
            pos = np.mean(positions, axis=0)
            self._obj_3d = pos
            self._trail.append(pos.copy())
            if len(self._trail) > GT_TRAIL_LEN:
                self._trail.pop(0)
