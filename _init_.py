"""
ADAS Perception Modules for YOLOPv2
====================================

A modular ADAS layer built on top of YOLOPv2 inference outputs.
No network modification, no retraining.

Modules:
    LaneTracker        — lane fitting, LDW, curvature, confidence
    SteeringEstimator  — pure-pursuit steering angle
    ObjectAnalyzer     — vehicle-in-lane, collision risk
    DrivableAreaAnalyzer — drivable area percentage
    ADASVisualizer     — professional dashboard + overlays

Usage:
    from adas.visualizer import ADASVisualizer
    adas = ADASVisualizer()
    results = adas.process_frame(img, da_mask, ll_mask, detections)
    rendered = adas.render(img, results, inf_time, fps, device)
"""
from .visualizer import ADASVisualizer
from .lane_tracker import LaneTracker
from .steering import SteeringEstimator
from .object_analysis import ObjectAnalyzer
from .drivable_area import DrivableAreaAnalyzer

__all__ = [
    'ADASVisualizer',
    'LaneTracker',
    'SteeringEstimator',
    'ObjectAnalyzer',
    'DrivableAreaAnalyzer',
]