"""
Steering Suggestion — pure-pursuit-inspired.
"""
import numpy as np
from .config import *
from .geometry import eval_poly, distance_to_image_y, smooth
from typing import Optional, Dict, Any

class SteeringEstimator:
    def __init__(self, img_h=720, img_w=1280):
        self.img_h, self.img_w = img_h, img_w
        self.vehicle_x = img_w / 2.0
        self.lookahead_m = LOOKAHEAD_M
        self.gain = 1.2
        self.smooth_a = 0.0
        self.alpha = 0.2  # Smoother steering wheel

    def estimate(self, center_coeffs, curvature_radius=float('inf')):
        if center_coeffs is None:
            return dict(angle_deg=0.0, direction='straight',
                        lookahead_x=self.vehicle_x, lookahead_y=self.img_h,
                        action='No lane data')

        ly = distance_to_image_y(self.lookahead_m, self.img_h)
        ly = max(self.img_h * 0.3, min(self.img_h - 1, ly))
        lx = float(eval_poly(center_coeffs, np.array([ly]))[0])

        off_m = (lx - self.vehicle_x) * XM_PER_PIX
        d_off = np.degrees(np.arctan2(off_m, self.lookahead_m))

        if not np.isinf(curvature_radius) and abs(curvature_radius) > 1:
            d_curve = np.degrees(np.arctan2(self.lookahead_m, abs(curvature_radius)))
            if curvature_radius < 0:
                d_curve = -d_curve
        else:
            d_curve = 0.0

        raw = self.gain * (0.6 * d_off + 0.4 * d_curve)
        raw = float(np.clip(raw, -MAX_STEER_DEG, MAX_STEER_DEG))
        self.smooth_a = smooth(raw, self.smooth_a, self.alpha)

        # Deadzone: if angle is tiny, force to exactly 0 to stop flickering
        if abs(self.smooth_a) < 1.5:
            self.smooth_a = 0.0

        if abs(self.smooth_a) < 1.0:
            d, act = 'straight', 'Keep Straight'
        elif self.smooth_a > 0:
            d, act = 'right', f'Steer Right {abs(self.smooth_a):.1f}°'
        else:
            d, act = 'left', f'Steer Left {abs(self.smooth_a):.1f}°'

        return dict(angle_deg=self.smooth_a, direction=d,
                    lookahead_x=lx, lookahead_y=float(ly), action=act)