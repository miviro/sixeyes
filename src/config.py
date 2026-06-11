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

# --- Mixture-of-Gaussians background subtraction ---
MOG_DELTA_FRAMES   = 200   # number of past frames used to build the background model
MOG_VAR_THRESHOLD  = 16    # squared Mahalanobis distance threshold for foreground classification
MOG_DETECT_SHADOWS = True

