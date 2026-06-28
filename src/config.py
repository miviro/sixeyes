# --- Display ---
CELL_W      = 640
CELL_H      = 360
TITLE_H     = 28
GAP         = 4
BG          = (30, 30, 30)
TITLE_BG    = (55, 55, 55)
TITLE_FG    = (220, 220, 220)
FONT_SCALE  = 0.55
FONT_THICK  = 1

# --- Camera ---
CAMERA_W    = 1920
CAMERA_H    = 1080
CAMERA_FPS  = 30

# --- Eighteyes ---
EIGHTEYES_MAX_INDEX = 16  # attempt connections eighteyes1..N.local

# --- Follow / servo tracking ---
FOLLOW_ZONE = 0.5          # dead-zone size as a fraction of each frame axis (centred)
FOLLOW_PORT = "/dev/ttyUSB0"

# --- Mixture-of-Gaussians background subtraction ---
MOG_DELTA_FRAMES   = 200   # number of past frames used to build the background model
MOG_VAR_THRESHOLD  = 16    # squared Mahalanobis distance threshold for foreground classification
MOG_DETECT_SHADOWS = True
MOG_SCALE_W        = 192   # downsample width before MOG (upscaled back afterwards)
MOG_SCALE_H        = 108   # downsample height before MOG

# --- 3-D ground truth reconstruction ---
# Intrinsics: OV2640 on AI Thinker ESP32-CAM at SVGA (800×600), M12 2.8 mm lens.
# Horizontal FOV ≈ 66°.  Source: sensor datasheet + common AI-Thinker lens spec.
GT_FOV_ESP32CAM       = 66.0   # degrees, horizontal
# Creative Live! Cam Sync 1080p: manufacturers quote 90° (diagonal).
GT_FOV_CREATIVE_1080P = 90.0   # degrees, diagonal
GT_FOV_DEFAULT        = 70.0   # fallback horizontal FOV for unknown sources

GT_BG_FRAMES   = 20     # quiet background frames to collect before pose estimation
GT_MIN_MATCHES = 25     # minimum ORB inlier matches for Essential Matrix estimation
GT_TRAIL_LEN   = 80     # object history trail length (frames)

GT_VIEW_W      = 640    # 3-D view panel width  — matches CELL_W so make_grid needs no rescale
GT_VIEW_H      = 360    # 3-D view panel height — matches CELL_H
GT_VIEW_FOCAL  = 380.0  # orbit-camera focal length (pixels) for the 3-D perspective view

