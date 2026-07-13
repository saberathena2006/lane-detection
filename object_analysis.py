"""
Object Analysis — vehicle-in-lane detection + collision risk.

Inputs
------
    detections    : list/array of [x1,y1,x2,y2,conf,cls]
    lane_polygon  : Nx2 int32 corridor polygon
    left_coeffs   : left lane polynomial
    right_coeffs  : right lane polynomial

Outputs
-------
    all_analyzed      : list of per-vehicle dicts
    in_lane_vehicles  : subset that are in current lane
    vehicle_ahead     : bool
    nearest_distance  : float (m)
    overall_risk      : 'SAFE' | 'CAUTION' | 'HIGH RISK'
    n_objects / n_vehicles : int

Algorithm
---------
1. Filter to COCO vehicle classes {2,5,7}.
2. For each vehicle, test bottom-center point against:
   a. Lane polygon (ray-casting).
   b. Polynomial bounds: left_x(y) ≤ bx ≤ right_x(y).
3. Estimate distance via pinhole: D = f·H·cosθ / (y_bottom − horizon).
4. Risk classification:
     not in lane            → SAFE
     D > SAFE_DIST          → SAFE
     D > CAUTION_DIST       → CAUTION
     D > HIGH_RISK_DIST     → CAUTION
     D ≤ HIGH_RISK_DIST     → HIGH RISK

Time Complexity: O(n·p), n=detections, p=polygon vertices.
Assumptions: bottom-center ≈ ground contact, flat road, constant speed.
Future improvements: stereo depth, optical-flow TTC, Kalman tracking.
"""
import numpy as np
from .config import *
from .geometry import point_in_polygon, image_y_to_distance
from typing import List, Dict, Any, Optional

RISK_SAFE    = 'SAFE'
RISK_CAUTION = 'CAUTION'
RISK_HIGH    = 'HIGH RISK'
RISK_COLORS  = {RISK_SAFE: C_SAFE, RISK_CAUTION: C_WARN, RISK_HIGH: C_CRIT}


class ObjectAnalyzer:
    def __init__(self, img_h=720, img_w=1280):
        self.img_h, self.img_w = img_h, img_w

    def analyze(self, detections, lane_polygon=None,
                left_coeffs=None, right_coeffs=None):
        dets = self._to_numpy(detections)
        vehicles = [d for d in dets if int(d[5]) in VEHICLE_CLASSES] if len(dets) else []

        analyzed, in_lane_veh = [], []
        nearest, best_risk, best_score = float('inf'), RISK_SAFE, 0

        for det in vehicles:
            in_lane = self._in_lane(det, lane_polygon, left_coeffs, right_coeffs)
            risk = self._risk(det, in_lane)
            entry = dict(detection=det, in_lane=in_lane, **risk)
            analyzed.append(entry)
            if in_lane:
                in_lane_veh.append(entry)
                if risk['distance_m'] < nearest:
                    nearest = risk['distance_m']
                if risk['risk_score'] > best_score:
                    best_score, best_risk = risk['risk_score'], risk['risk_level']

        return dict(
            all_analyzed=analyzed, in_lane_vehicles=in_lane_veh,
            vehicle_ahead=len(in_lane_veh) > 0,
            nearest_distance=nearest,
            overall_risk=best_risk if in_lane_veh else RISK_SAFE,
            n_objects=len(dets), n_vehicles=len(vehicles),
        )

    # ── internal ─────────────────────────────────────────────────────

    @staticmethod
    def _to_numpy(dets):
        import torch
        if isinstance(dets, torch.Tensor):
            dets = dets.cpu().numpy()
        dets = np.asarray(dets)
        if dets.size == 0:
            return dets.reshape(0, 6)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        return dets[:, :6] if dets.shape[1] >= 6 else dets

    def _in_lane(self, det, polygon, lc, rc):
        x1, y1, x2, y2 = det[:4]
        bx, by = (x1 + x2) / 2, y2
        # polygon test
        in_poly = False
        if polygon is not None and len(polygon) >= 3:
            in_poly = point_in_polygon((bx, by), polygon)
        # polynomial test
        if lc is not None and rc is not None:
            lx = float(np.polyval(lc, by))
            rx = float(np.polyval(rc, by))
            lo, hi = min(lx, rx) - 30, max(lx, rx) + 30
            return (lo <= bx <= hi) or in_poly
        return in_poly

    @staticmethod
    def _risk(det, in_lane):
        x1, y1, x2, y2 = det[:4]
        dist = image_y_to_distance(float(y2))
        area = max(1, (x2 - x1) * (y2 - y1))
        if not in_lane:
            return dict(distance_m=dist, risk_level=RISK_SAFE,
                        risk_score=0, bbox_area=area)
        if np.isinf(dist) or dist > SAFE_DIST:
            rl, rs = RISK_SAFE, max(0, 30 - dist) if not np.isinf(dist) else 0
        elif dist > CAUTION_DIST:
            rl, rs = RISK_CAUTION, 30 + (CAUTION_DIST - dist) / CAUTION_DIST * 40
        elif dist > HIGH_RISK_DIST:
            rl, rs = RISK_CAUTION, 60 + (CAUTION_DIST - dist) / CAUTION_DIST * 20
        else:
            rl, rs = RISK_HIGH, 80 + (HIGH_RISK_DIST - dist) / HIGH_RISK_DIST * 20
        return dict(distance_m=dist, risk_level=rl,
                    risk_score=min(100, max(0, rs)), bbox_area=area)