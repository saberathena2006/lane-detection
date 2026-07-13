"""
Professional ADAS Visualizer — dashboard + all overlays.
"""
import numpy as np
import cv2
from .config import *
from .geometry import compute_distance_bands, eval_poly
from .lane_tracker import LaneTracker
from .steering import SteeringEstimator
from .object_analysis import ObjectAnalyzer, RISK_COLORS, RISK_SAFE
from .drivable_area import DrivableAreaAnalyzer

class ADASVisualizer:
    def __init__(self, img_h=720, img_w=1280):
        self.h, self.w = img_h, img_w
        self.lane = LaneTracker(img_h, img_w)
        self.steer = SteeringEstimator(img_h, img_w)
        self.obj = ObjectAnalyzer(img_h, img_w)
        self.da = DrivableAreaAnalyzer(img_h, img_w)

    def process_frame(self, im0, da_mask, ll_mask, detections):
        h, w = im0.shape[:2]
        if da_mask.shape[:2] != (h, w):
            da_mask = cv2.resize(da_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        if ll_mask.shape[:2] != (h, w):
            ll_mask = cv2.resize(ll_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        lane_res = self.lane.update(ll_mask)
        steer_res = self.steer.estimate(lane_res['center_coeffs'], lane_res['curvature']['radius_m'])
        obj_res = self.obj.analyze(
            detections,
            lane_polygon=lane_res['lane_polygon'],
            left_coeffs=lane_res['left_coeffs'],
            right_coeffs=lane_res['right_coeffs'])
        da_res = self.da.analyze(da_mask)

        return dict(lane=lane_res, steering=steer_res, objects=obj_res, drivable_area=da_res, da_mask=da_mask, ll_mask=ll_mask)

    def render(self, im0, results, inf_time=0.0, fps=0.0, device='CPU'):
        img = im0.copy()
        overlay = img.copy()

        self._draw_drivable(overlay, results.get('da_mask'))
        self._draw_lane_mask(overlay, results.get('ll_mask'))  # Draw raw model mask like original demo
        self._draw_corridor(overlay, results['lane'])
        self._draw_dist_bands(overlay)
        self._draw_lane_lines(overlay, results['lane'])
        self._draw_trajectory(overlay, results['lane'])
        self._draw_objects(overlay, results['objects'])
        self._draw_steer_arrow(overlay, results['steering'])
        self._draw_ldw(overlay, results['lane'])

        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

        self._draw_curvature(img, results['lane'])
        self._draw_dashboard(img, results, inf_time, fps, device)
        return img

    def _draw_drivable(self, img, da_mask):
        if da_mask is None or not np.any(da_mask): return
        tint = img.copy()
        tint[da_mask > 0] = C_DA
        cv2.addWeighted(tint, 0.4, img, 0.6, 0, img)

    def _draw_lane_mask(self, img, ll_mask):
        """Draws the exact model output mask so it never spills."""
        if ll_mask is None or not np.any(ll_mask): return
        tint = img.copy()
        tint[ll_mask > 0] = C_LANE
        cv2.addWeighted(tint, 0.6, img, 0.4, 0, img)

    def _draw_corridor(self, img, lane):
        poly = lane.get('lane_polygon')
        if poly is None or len(poly) < 3: return
        dep = lane.get('departure', {})
        if dep.get('critical'):
            color = C_CORRIDOR_X
        elif dep.get('warning'):
            color = C_WARN
        else:
            color = C_CORRIDOR
        cv2.fillPoly(img, [poly], color)

    def _draw_dist_bands(self, img):
        for d, y in compute_distance_bands(self.h):
            cv2.line(img, (0, int(y)), (self.w, int(y)), C_DIST, 1, cv2.LINE_AA)
            cv2.putText(img, f'{d}m', (self.w - 60, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_DIST, 1, cv2.LINE_AA)

    def _draw_lane_lines(self, img, lane):
        safe_top = lane.get('safe_y_top', self.h * 0.55)
        ys = np.linspace(safe_top, self.h - 1, 50)
        for coeffs, color in [(lane.get('left_coeffs'), (255, 255, 0)), (lane.get('right_coeffs'), (255, 255, 0))]:
            if coeffs is not None:
                xs = eval_poly(coeffs, ys)
                pts = np.stack([xs, ys], 1).astype(np.int32)
                cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)

    def _draw_trajectory(self, img, lane):
        cl = lane.get('centerline_points')
        if cl is None or len(cl) < 2: return
        pts = cl.astype(np.int32)
        cv2.polylines(img, [pts], False, (0, 100, 0), 7, cv2.LINE_AA)
        cv2.polylines(img, [pts], False, C_TRAJ, 3, cv2.LINE_AA)
        for i in range(0, len(pts), 5):
            cv2.circle(img, tuple(pts[i]), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(pts[i]), 3, C_TRAJ, -1, cv2.LINE_AA)

    def _draw_objects(self, img, objects):
        for e in objects.get('all_analyzed', []):
            det = e['detection']
            x1, y1, x2, y2 = [int(v) for v in det[:4]]
            in_lane = e['in_lane']
            rl = e['risk_level']
            dist = e['distance_m']

            color = RISK_COLORS.get(rl, C_CRIT) if in_lane else C_SAFE
            thick = 3 if in_lane else 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)

            parts = []
            if in_lane: parts.append('IN LANE')
            parts.append(rl)
            if not np.isinf(dist): parts.append(f'{dist:.1f}m')
            label = ' | '.join(parts)
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            if in_lane and not np.isinf(dist) and dist < 30:
                cv2.putText(img, 'Vehicle Ahead', (int((x1+x2)/2) - 60, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_CRIT, 2, cv2.LINE_AA)

    def _draw_steer_arrow(self, img, st):
        angle = st.get('angle_deg', 0)
        cx, cy = self.w // 2, self.h - 80
        length = 70
        rad = np.radians(angle)
        ex = cx + int(length * np.sin(rad))
        ey = cy - int(length * np.cos(rad))
        color = C_SAFE if abs(angle) < 2 else C_WARN if abs(angle) < 10 else C_CRIT
        cv2.arrowedLine(img, (cx, cy), (ex, ey), color, 4, cv2.LINE_AA, 0, 0.3)
        cv2.putText(img, f'{angle:+.1f}°', (cx - 30, cy + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    def _draw_ldw(self, img, lane):
        dep = lane.get('departure', {})
        if not dep.get('warning'): return
        d = dep.get('direction', 'none')
        crit = dep.get('critical')
        text = '!! LANE DEPARTURE !!' if crit else 'LANE DEPARTURE WARNING'
        color = C_CRIT if crit else C_WARN
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        x = (self.w - tw) // 2
        y = 55
        cv2.rectangle(img, (x - 12, y - th - 10), (x + tw + 12, y + 10), (0, 0, 0), -1, cv2.LINE_AA)
        cv2.rectangle(img, (x - 12, y - th - 10), (x + tw + 12, y + 10), color, 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        if d != 'none':
            ax = x + tw + 35 if d == 'right' else x - 35
            dx = 30 if d == 'right' else -30
            cv2.arrowedLine(img, (ax, y - th // 2), (ax + dx, y - th // 2), color, 3, cv2.LINE_AA, 0, 0.5)

    def _draw_curvature(self, img, lane):
        curv = lane.get('curvature', {})
        rt = curv.get('road_type', 'Unknown')
        r = curv.get('radius_m', float('inf'))
        x, y = self.w - 230, 60
        cv2.putText(img, f'Road: {rt}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_INFO, 1, cv2.LINE_AA)
        r_text = 'Radius: --' if np.isinf(r) else f'Radius: {abs(r):.0f}m'
        cv2.putText(img, r_text, (x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_INFO, 1, cv2.LINE_AA)

    def _draw_dashboard(self, img, res, inf_t, fps, device):
        lane, st, obj, da = res['lane'], res['steering'], res['objects'], res['drivable_area']
        dw = DASH_W
        cv2.rectangle(img, (0, 0), (dw, self.h), C_BG, -1)
        cv2.rectangle(img, (0, 0), (dw, self.h), C_BORDER, 1)

        cv2.putText(img, 'ADAS PERCEPTION', (DASH_PAD, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_ACCENT, 2, cv2.LINE_AA)
        cv2.line(img, (10, 38), (dw - 10, 38), C_BORDER, 1, cv2.LINE_AA)

        rows = []
        rows.append(('FPS', f'{fps:.1f}', C_TEXT))
        rows.append(('Inference', f'{inf_t*1000:.1f} ms', C_TEXT))
        rows.append(('─'*22, '', C_BORDER))

        dep = lane.get('departure', {})
        if dep.get('critical'): ls, lc = 'DEPARTURE!', C_CRIT
        elif dep.get('warning'): ls, lc = 'DRIFTING', C_WARN
        elif lane.get('left_found') and lane.get('right_found'): ls, lc = 'CENTERED', C_SAFE
        else: ls, lc = 'NO LANE', C_WARN
        rows.append(('Lane Status', ls, lc))
        rows.append(('Road Type', lane.get('curvature', {}).get('road_type', '?'), C_TEXT))

        sa = st.get('angle_deg', 0)
        sc = C_SAFE if abs(sa) < 5 else C_WARN
        rows.append(('Steering', f'{sa:+.1f}°', sc))

        off = lane.get('offset_m', 0)
        oc = C_SAFE if abs(off) < 0.5 else C_WARN
        rows.append(('Lane Offset', f'{off:+.2f} m', oc))

        conf = lane.get('confidence', 0)
        cc = C_SAFE if conf > 60 else C_WARN if conf > 30 else C_CRIT
        rows.append(('Lane Conf.', f'{conf:.0f}%', cc))

        rows.append(('Drivable', f'{da.get("percentage",0):.1f}%', C_TEXT))
        rows.append(('─'*22, '', C_BORDER))
        rows.append(('Objects', f'{obj.get("n_objects",0)}', C_TEXT))
        rows.append(('Vehicles', f'{obj.get("n_vehicles",0)}', C_TEXT))

        va = obj.get('vehicle_ahead', False)
        rows.append(('Veh. Ahead', 'YES' if va else 'NO', C_CRIT if va else C_SAFE))

        nd = obj.get('nearest_distance', float('inf'))
        if np.isinf(nd): ndt, ndc = '--', C_TEXT
        else:
            ndt = f'{nd:.1f} m'
            ndc = C_CRIT if nd < HIGH_RISK_DIST else C_WARN if nd < CAUTION_DIST else C_SAFE
        rows.append(('Nearest', ndt, ndc))

        risk = obj.get('overall_risk', RISK_SAFE)
        rows.append(('Collision', risk, RISK_COLORS.get(risk, C_TEXT)))
        rows.append(('─'*22, '', C_BORDER))
        rows.append(('Compute', device, C_TEXT))

        y = 62
        for label, value, color in rows:
            if label.startswith('─'):
                cv2.line(img, (10, y + 4), (dw - 10, y + 4), C_BORDER, 1, cv2.LINE_AA)
                y += 14
                continue
            cv2.putText(img, label, (DASH_PAD, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_TEXT_DIM, 1, cv2.LINE_AA)
            (vw, _), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.putText(img, value, (dw - DASH_PAD - vw, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
            y += DASH_LINE_H

        by = self.h - 32
        cv2.rectangle(img, (10, by), (dw - 10, by + 22), (35, 35, 38), -1)
        rc = RISK_COLORS.get(risk, C_SAFE)
        cv2.circle(img, (22, by + 11), 5, rc, -1, cv2.LINE_AA)
        cv2.putText(img, 'SYSTEM ACTIVE', (36, by + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_TEXT, 1, cv2.LINE_AA)