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
HFOV_DEG    = 69.3   # Creative Live! Cam Sync 1080p horizontal FOV (degrees)
VFOV_DEG    = 42.4   # Creative Live! Cam Sync 1080p vertical FOV (degrees)

# --- Eighteyes ---
EIGHTEYES_MAX_INDEX = 16  # attempt connections eighteyes1..N.local

# --- Follow / servo tracking ---
FOLLOW_ZONE     = 0.5          # dead-zone size as a fraction of each frame axis (centred)
FOLLOW_PORT     = "/dev/ttyUSB0"
SWEEP_PATIENCE  = 30           # frames without any detection before sweep starts
DZ_COLOR        = (0, 220, 0)  # dead-zone rectangle colour (BGR)

# --- Main loop ---
EMA_ALPHA = 0.05  # smoothing factor for frame-time EMA

# --- Mixture-of-Gaussians background subtraction ---
MOG_DELTA_FRAMES   = 200   # number of past frames used to build the background model
MOG_VAR_THRESHOLD  = 16    # squared Mahalanobis distance threshold for foreground classification
MOG_DETECT_SHADOWS = True
MOG_SCALE_W        = 192   # downsample width before MOG (upscaled back afterwards)
MOG_SCALE_H        = 108   # downsample height before MOG

