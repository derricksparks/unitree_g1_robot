"""Vision utilities for G1 MuJoCo demos."""

from .red_box_detector import RedBoxCameraDetector, RedBoxDetectionResult, detect_red_box_from_camera

__all__ = [
    "RedBoxCameraDetector",
    "RedBoxDetectionResult",
    "detect_red_box_from_camera",
]

