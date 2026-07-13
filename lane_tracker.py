"""
Lane Tracking — detection, LDW, curvature, confidence.
"""
import numpy as np
import cv2
from .config import *
from .geometry import fit_poly, eval_poly, curvature_radius, smooth, _horizon_y
from typing import Optional, Dict, Any

class LaneTracker:
    def __init__(self, img_h=720, img_w=1280):
        self.h, self.w = img_h, img_w
        self.y_bot = img_h - 1
        self.horizon = _horizon_y()
        self.y_top = max(int(img_h * 0.60), int(self.horizon + 40))
        self.vehicle_x = img_w / 2.0
        self.alpha = 0.2
        self.smoothed_ar = 2000.0 
        self.left_coeffs: Optional[np.ndarray] = None
        self.right_coeffs: Optional[np.ndarray] = None
        self.bot_width = LANE_WIDTH_PX  # Dynamic width based on bottom pixels

    def update(self, lane_mask: np.ndarray) -> Dict[str, Any]:
        mask = (lane_mask > 0).astype(np.uint8)
        if mask.shape[:2] != (self.h, self.w):
            mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        ld = self._detect(mask)
        center = self._center(ld['left_coeffs'], ld['right_coeffs'])
        off_px, off_m = self._offset(center)
        dep = self._departure(off_m)
        curve_c = center if center is not None else (ld['left_coeffs'] or ld['right_coeffs'])
        curv = self._curvature(curve_c)
        conf = self._confidence(ld)
        
        # Calculate dynamic width at the bottom
        if ld['left_coeffs'] is not None and ld['right_coeffs'] is not None:
            lx_bot = eval_poly(ld['left_coeffs'], np.array([float(self.y_bot)]))[0]
            rx_bot = eval_poly(ld['right_coeffs'], np.array([float(self.y_bot)]))[0]
            self.bot_width = 0.9 * self.bot_width + 0.1 * max(300.0, (rx_bot - lx_bot))
            
        safe_y_top = self._calculate_safe_top(ld['left_coeffs'], ld['right_coeffs'])
        cl_pts = self._centerline_points(center, safe_y_top)
        polygon = self._polygon(ld['left_coeffs'], ld['right_coeffs'], safe_y_top)

        return dict(
            left_coeffs=ld['left_coeffs'], right_coeffs=ld['right_coeffs'],
            center_coeffs=center, offset_px=off_px, offset_m=off_m,
            departure=dep, curvature=curv, confidence=conf,
            centerline_points=cl_pts, lane_polygon=polygon,
            left_found=ld['left_found'], right_found=ld['right_found'],
            safe_y_top=safe_y_top
        )

    def _calculate_safe_top(self, lc, rc):
        """Find the highest point where the lane width is physically valid."""
        if lc is None or rc is None:
            return float(self.h * 0.65)
            
        # Scan from bottom to top to find where lines diverge or cross
        safe_y = self.y_bot
        for y in range(self.y_bot, int(self.h * 0.5), -5):
            lx = eval_poly(lc, np.array([float(y)]))[0]
            rx = eval_poly(rc, np.array([float(y)]))[0]
            width = rx - lx
            
            # If width gets too narrow or too wide compared to bottom, cut it off
            if width < self.bot_width * 0.4 or width > self.bot_width * 1.5 or rx <= lx:
                safe_y = y + 5  # Step back one iteration
                break
            safe_y = y
            
        # Never allow the safe top to go above 60% of the screen to avoid horizon noise
        return float(max(safe_y, self.h * 0.60))

    def _detect(self, mask):
        h, w = mask.shape[:2]
        hist = np.sum(mask[h // 2:, :], axis=0)
        mid = w // 2

        lp = np.argmax(hist[:mid]) if hist[:mid].max() > 5 else None
        rp = mid + np.argmax(hist[mid:]) if hist[mid:].max() > 5 else None

        if lp is None and self.left_coeffs is not None:
            lp = int(eval_poly(self.left_coeffs, np.array([self.y_bot]))[0])
        if rp is None and self.right_coeffs is not None:
            rp = int(eval_poly(self.right_coeffs, np.array([self.y_bot]))[0])

        lx, ly = self._slide(mask, lp) if lp is not None else ([], [])
        rx, ry = self._slide(mask, rp) if rp is not None else ([], [])

        lc = fit_poly(lx, ly) if len(lx) >= 3 else None
        rc = fit_poly(rx, ry) if len(rx) >= 3 else None

        if lc is not None and rc is None:
            rc = lc.copy(); rc[2] += LANE_WIDTH_PX
        elif rc is not None and lc is None:
            lc = rc.copy(); lc[2] -= LANE_WIDTH_PX

        lc = self._smooth_side(lc, 'left')
        rc = self._smooth_side(rc, 'right')

        return dict(left_coeffs=lc, right_coeffs=rc,
                    left_found=lc is not None, right_found=rc is not None,
                    left_pixels=(np.array(lx), np.array(ly)),
                    right_pixels=(np.array(rx), np.array(ry)),
                    histogram=hist)

    def _slide(self, mask, x_start):
        h, w = mask.shape[:2]
        wh = max(1, (self.y_bot - self.y_top) // N_WINDOWS)
        xc = x_start
        pxs, pys = [], []
        nzy, nzx = mask.nonzero()
        
        for win in range(N_WINDOWS):
            yb = self.y_bot - win * wh
            yt = yb - wh
            xl = max(0, xc - 100) 
            xr = min(w, xc + 100)
            sel = ((nzy >= yt) & (nzy < yb) & (nzx >= xl) & (nzx < xr))
            gx, gy = nzx[sel], nzy[sel]
            pxs.extend(gx.tolist()); pys.extend(gy.tolist())
            if len(gx) >= 20:
                xc = int(np.mean(gx))
        return pxs, pys

    def _smooth_side(self, current, side):
        prev = self.left_coeffs if side == 'left' else self.right_coeffs
        if current is not None and prev is not None:
            sm = smooth(current, prev, self.alpha)
        else:
            sm = current if current is not None else prev
        if side == 'left':
            self.left_coeffs = sm
        else:
            self.right_coeffs = sm
        return sm

    @staticmethod
    def _center(lc, rc):
        if lc is None and rc is None: return None
        if lc is None: return rc
        if rc is None: return lc
        return (lc + rc) / 2.0

    def _offset(self, center):
        if center is None: return 0.0, 0.0
        cx = eval_poly(center, np.array([float(self.y_bot)]))[0]
        off_px = self.vehicle_x - cx
        return float(off_px), float(off_px * XM_PER_PIX)

    @staticmethod
    def _departure(off_m):
        hw = LANE_WIDTH_M / 2
        norm = abs(off_m) / hw if hw > 0 else 0
        warn = norm > LDW_THRESHOLD
        crit = norm > LDW_CRITICAL
        d = 'right' if off_m > 0 and warn else 'left' if off_m < 0 and warn else 'none'
        return dict(warning=warn, critical=crit, direction=d, normalized_offset=norm)

    def _curvature(self, coeffs):
        if coeffs is None:
            return dict(radius_m=float('inf'), road_type='Unknown', direction='unknown')
        
        y_eval = 680.0 
        r = curvature_radius(coeffs, y_eval)
        ar = abs(r)
        
        if ar > 1500:
            ar = 3000.0 
            
        self.smoothed_ar = 0.15 * ar + 0.85 * self.smoothed_ar
        ar = self.smoothed_ar

        top_x = float(eval_poly(coeffs, np.array([450.0]))[0])
        bot_x = float(eval_poly(coeffs, np.array([719.0]))[0])
        direction = 'right' if top_x > bot_x else 'left' if top_x < bot_x else 'straight'

        if ar > CURVE_STRAIGHT:
            rt, d = 'Straight', 'straight'
        elif ar > CURVE_GENTLE:
            rt = f'Gentle {direction.capitalize()}'
            d = direction
        else:
            rt = f'Sharp {direction.capitalize()}'
            d = direction

        r_signed = ar if direction == 'right' else -ar if direction == 'left' else float('inf')
        return dict(radius_m=r_signed, road_type=rt, direction=d)

    def _confidence(self, ld):
        lf, rf = ld['left_found'], ld['right_found']
        if not lf and not rf: return 0.0
        s = 0.0
        s += 30 if lf and rf else 15
        total_px = len(ld['left_pixels'][0]) + len(ld['right_pixels'][0])
        s += min(30, total_px / 50.0)
        all_y = list(ld['left_pixels'][1]) + list(ld['right_pixels'][1])
        if all_y:
            s += min(20, (max(all_y) - min(all_y)) / self.h * 40)
        hist = ld.get('histogram', np.array([]))
        if hist.size > 0:
            s += min(20, hist.max() / (self.h / 2.0) * 20)
        return float(min(100, max(0, s)))

    def _centerline_points(self, center, safe_y_top, n=50):
        if center is None: return np.array([])
        ys = np.linspace(safe_y_top, self.y_bot, n)
        xs = eval_poly(center, ys)
        return np.stack([xs, ys], axis=1)

    def _polygon(self, lc, rc, safe_y_top, n=30):
        if lc is None and rc is None: return np.array([])
        ys = np.linspace(safe_y_top, self.y_bot, n)
        lx = eval_poly(lc, ys) if lc is not None else None
        rx = eval_poly(rc, ys) if rc is not None else None
        if lx is None: lx = rx - self.bot_width
        if rx is None: rx = lx + self.bot_width
            
        pts = np.vstack([np.stack([lx, ys], 1), np.stack([rx, ys], 1)[::-1]])
        return pts.astype(np.int32)