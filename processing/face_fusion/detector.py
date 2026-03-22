"""Face detection — M12 internal per Video Pipeline v3.2.0 freeze spec."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from video_pipeline import ROIBox


class FaceDetector:
    """Detects faces in video frames and returns bounding boxes.

    Requires a detection model backend to operate.
    """

    def __init__(self, min_confidence: float = 0.5) -> None:
        raise NotImplementedError(
            "FaceDetector.__init__: real implementation requires a face "
            "detection model backend (e.g. RetinaFace / ONNX runtime). "
            "This module is declared in the M12 freeze spec but has no "
            "production implementation yet."
        )

    def detect(self, frame: np.ndarray) -> list[ROIBox]:
        raise NotImplementedError(
            "FaceDetector.detect: no production implementation available. "
            "Expected to accept an (H, W, 3) uint8 BGR frame and return "
            "a list of ROIBox detections."
        )
