"""FaceFusionConfig DTO — DM-09 per Video Pipeline v3.2.0 freeze spec."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaceFusionConfig:
    enabled: bool
    mode: str
    reference_face_path: str
    strength: float
    min_confidence: float
    preserve_background: bool
