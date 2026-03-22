"""Face fusion engine — M12 per Video Pipeline v3.2.0 freeze spec."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from dto.face_fusion import FaceFusionConfig


class FaceFusionEngine:
    """Applies face-fusion transformations to video frames.

    Requires a trained model and reference face embeddings to operate.
    """

    def __init__(self, config: FaceFusionConfig) -> None:
        raise NotImplementedError(
            "FaceFusionEngine.__init__: real implementation requires "
            "a face-fusion model backend (e.g. InsightFace / ONNX runtime). "
            "This module is declared in the M12 freeze spec but has no "
            "production implementation yet."
        )

    def fuse(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "FaceFusionEngine.fuse: no production implementation available. "
            "Expected to accept an (H, W, 3) uint8 BGR frame and return "
            "the fused result."
        )

    def reset(self) -> None:
        raise NotImplementedError(
            "FaceFusionEngine.reset: no production implementation available."
        )
