"""
Configuration constants for all ADAS modules.

All distances in meters, angles in degrees, pixels in image space.
Values assume a typical forward-facing dashcam (~1.5 m height, ~5° pitch).
Tune these for your specific camera setup.
"""
import numpy as np

# ── Camera model (pinhole, flat-ground) ──────────────────────────────
CAMERA_HEIGHT    = 1.5       # m above ground
CAMERA_PITCH     = 5.0       # deg downward tilt
FOCAL_LENGTH     = 700.0     # px (for 1280×720)
IMAGE_CENTER_X   = 640.0
IMAGE_CENTER_Y   = 360.0

# ── Lane geometry ────────────────────────────────────────────────────
LANE_WIDTH_M     = 3.7       # standard lane width (m)
LANE_WIDTH_PX    = 700.0     # approx lane width in px at image bottom
YM_PER_PIX       = 30.0 / 720.0   # m per px (vertical)
XM_PER_PIX       = 3.7 / 700.0    # m per px (horizontal)

# ── Lane Departure Warning ───────────────────────────────────────────
LDW_THRESHOLD    = 0.30      # |offset| / (half-lane-width) > this → warn
LDW_CRITICAL     = 0.65      # > this → critical

# ── Sliding-window lane search ───────────────────────────────────────
N_WINDOWS        = 9
WINDOW_MARGIN    = 50        # px (half-width)
MIN_PIX_PER_WIN  = 50

# ── Curvature classification (radius in m) ───────────────────────────
CURVE_STRAIGHT   = 1000.0
CURVE_GENTLE     = 500.0

# ── Steering ─────────────────────────────────────────────────────────
LOOKAHEAD_M      = 15.0
MAX_STEER_DEG    = 30.0

# ── Collision risk distance thresholds (m) ───────────────────────────
SAFE_DIST        = 30.0
CAUTION_DIST     = 15.0
HIGH_RISK_DIST   = 8.0

# ── Distance bands to draw (m) ───────────────────────────────────────
DISTANCE_BANDS   = [5, 10, 15, 20, 30, 50]

# ── COCO vehicle classes ─────────────────────────────────────────────
VEHICLE_CLASSES  = {2, 5, 7}   # car, bus, truck

# ── Colors (BGR) ─────────────────────────────────────────────────────
C_SAFE      = (0, 200, 0)
C_WARN      = (0, 200, 255)
C_CRIT      = (0, 0, 255)
C_INFO      = (255, 255, 255)
C_LANE      = (0, 255, 255)
C_TRAJ      = (0, 255, 0)
C_CORRIDOR  = (0, 180, 0)
C_CORRIDOR_X= (0, 0, 180)
C_DIST      = (180, 180, 180)
C_DA        = (0, 80, 0)       # drivable area tint
C_BG        = (18, 18, 20)
C_BORDER    = (55, 55, 60)
C_TEXT      = (235, 235, 235)
C_TEXT_DIM  = (150, 150, 160)
C_ACCENT    = (0, 220, 220)

# ── Dashboard layout ─────────────────────────────────────────────────
DASH_W       = 300
DASH_ALPHA   = 0.88
DASH_PAD     = 12
DASH_LINE_H  = 26