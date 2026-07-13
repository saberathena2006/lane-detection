"""
Drivable Area Analytics.

Input:  da_mask — HxW binary (1 = drivable)
Output: percentage of drivable pixels in the lower 60 % of the image.

The lower 60 % is used as the road region-of-interest to exclude sky
and distant scenery, giving a more meaningful percentage.

Time Complexity: O(H·W) single-pass sum.
Assumptions: mask is roughly aligned with image; road is in lower portion.
Future improvements: dynamic ROI based on vanishing point, free-space depth.
"""
import numpy as np
from typing import Dict, Any


class DrivableAreaAnalyzer:
    def __init__(self, img_h=720, img_w=1280):
        self.roi_top = int(img_h * 0.4)

    def analyze(self, da_mask: np.ndarray) -> Dict[str, Any]:
        mask = (da_mask > 0).astype(np.uint8)
        roi = mask[self.roi_top:, :]
        total = roi.size
        drv = int(np.sum(roi))
        pct = (drv / total * 100.0) if total > 0 else 0.0
        return dict(percentage=pct, pixel_count=drv,
                    total_pixels=total, has_drivable_area=drv > 0)