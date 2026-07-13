"""
Geometry & perspective utilities for ADAS.

Provides:
    - image_y ↔ ground-distance conversion (pinhole + flat ground)
    - pixel ↔ world-lateral conversion
    - polynomial fitting x = f(y) for lane lines
    - curvature radius computation
    - point-in-polygon test (ray casting)
    - exponential-moving-average smoothing

Mathematical Approach
---------------------
Pinhole camera at height H, pitch θ (downward).  A ground point at
distance D, lateral offset X projects to image (u,v):

    u = cx + f·X / (D·cosθ + H·sinθ)
    v = cy + f·(H·cosθ − D·sinθ) / (D·cosθ + H·sinθ)

For small θ and H≪D  →  v ≈ cy − f·tanθ + f·H / D
Inverse:  D = f·H·cosθ / (v − (cy − f·tanθ))

Curvature (after converting px→m):
    x = a·y² + b·y + c  (pixels)
    X = A·Y² + B·Y + C  (meters)
    A = xm·a / ym²,  B = xm·b / ym
    R = (1 + (2AY+B)²)^{3/2} / |2A|

Time Complexity: O(n) for conversions, O(n·d²) for polyfit (d=2).
Assumptions: flat road, constant pitch, no roll, no lens distortion.
Future improvements: stereo depth, IMU pitch fusion, distortion correction.
"""
import numpy as np
import cv2
from .config import *


# ── Perspective / distance ───────────────────────────────────────────

def _horizon_y() -> float:
    """Image y-coordinate of the horizon (vanishing line)."""
    return IMAGE_CENTER_Y - FOCAL_LENGTH * np.tan(np.radians(CAMERA_PITCH))


def image_y_to_distance(y: float) -> float:
    """Convert image y (px) → ground distance (m).  Returns inf at/above horizon."""
    dy = y - _horizon_y()
    if dy < 1:
        return float('inf')
    return FOCAL_LENGTH * CAMERA_HEIGHT * np.cos(np.radians(CAMERA_PITCH)) / dy


def distance_to_image_y(d: float, img_h: int = 720) -> int:
    """Convert ground distance (m) → image y (px)."""
    h = _horizon_y()
    y = h + FOCAL_LENGTH * CAMERA_HEIGHT * np.cos(np.radians(CAMERA_PITCH)) / max(d, 0.1)
    return int(np.clip(y, 0, img_h - 1))


def pixel_to_world_x(x: float, y: float) -> float:
    """Image (x,y) → lateral world offset (m, +right)."""
    dist = image_y_to_distance(y)
    if np.isinf(dist):
        return 0.0
    return (x - IMAGE_CENTER_X) * dist / FOCAL_LENGTH


# ── Polynomial helpers ───────────────────────────────────────────────

def fit_poly(xs, ys, deg=2):
    """Least-squares fit x = f(y).  Returns [a,b,c] or None."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if len(xs) < deg + 1:
        return None
    try:
        return np.polyfit(ys, xs, deg)
    except (np.linalg.LinAlgError, ValueError):
        return None


def eval_poly(coeffs, ys):
    """Evaluate polynomial at ys."""
    return np.polyval(coeffs, ys)


def curvature_radius(coeffs, y_eval):
    """Curvature radius (m) at y_eval.  +→right curve, −→left, inf→straight."""
    if coeffs is None or len(coeffs) < 3:
        return float('inf')
    A = coeffs[0] * XM_PER_PIX / (YM_PER_PIX ** 2)
    B = coeffs[1] * XM_PER_PIX / YM_PER_PIX
    Y = y_eval * YM_PER_PIX
    num = (1 + (2 * A * Y + B) ** 2) ** 1.5
    den = abs(2 * A)
    if den < 1e-10:
        return float('inf')
    return np.sign(A) * num / den


# ── Geometry ─────────────────────────────────────────────────────────

def point_in_polygon(pt, poly):
    """Ray-casting point-in-polygon test.  pt=(x,y), poly=Nx2."""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def smooth(current, previous, alpha=0.3):
    """Exponential moving average.  alpha=1 → no smoothing."""
    if previous is None:
        return current
    if isinstance(current, np.ndarray):
        return alpha * current + (1 - alpha) * previous
    return alpha * current + (1 - alpha) * previous


# ── Distance bands ───────────────────────────────────────────────────

def compute_distance_bands(img_h=720):
    """Return [(distance_m, y_px), ...] for DISTANCE_BANDS."""
    return [(d, distance_to_image_y(d, img_h)) for d in DISTANCE_BANDS
            if 0 < distance_to_image_y(d, img_h) < img_h]


def bbox_distance(bbox, img_h=720):
    """Estimate distance (m) from bbox bottom-center (ground contact)."""
    return image_y_to_distance(bbox[3])