"""
video_pipeline.py — Video Pipeline v3.2.0
DTOs (M2), ROI (M4), OCR (M5), SubtitleTracker (M6), TranslationBuffer (M7),
TranslationCache/Adapter (M8/M9), RecolorEngine (M11), FaceFusionEngine (M12),
SubtitleRenderer (M13), FPSNormalizer (M14), ReaderThread (M15),
ProcessorThread (M16), EncoderThread (M17), BackboneExecutionCore (M18),
VideoPipeline (M19).
"""
from __future__ import annotations

import hashlib
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from config_loader import stage_timer

try:
    from dto.face_fusion import FaceFusionConfig
    from processing.face_fusion.detector import FaceDetector
    from processing.face_fusion.engine import FaceFusionEngine
    from processing.face_fusion.exceptions import FaceFusionInputError
except ImportError:  # pragma: no cover
    from video_pipeline.dto.face_fusion import FaceFusionConfig
    from video_pipeline.processing.face_fusion.detector import FaceDetector
    from video_pipeline.processing.face_fusion.engine import FaceFusionEngine
    from video_pipeline.processing.face_fusion.exceptions import FaceFusionInputError


SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
ENGINE_MODES = {"HSV", "LAB"}
FACE_FUSION_ALLOWED_MODES = {"single", "first", "top1", "one", "multi", "all"}


def _validate_tuple_range(value: Tuple[int, int], name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be tuple[int, int]")
    a, b = value
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError(f"{name} values must be int")
    if a < 0 or b < 0 or a > 255 or b > 255:
        raise ValueError(f"{name} values must be within range 0-255")
    if a > b:
        raise ValueError(f"{name} start must be <= end")


def _validate_face_fusion_config(cfg: FaceFusionConfig) -> None:
    if not isinstance(cfg, FaceFusionConfig):
        raise FaceFusionInputError("Invalid FaceFusionConfig")
    if not isinstance(cfg.enabled, bool):
        raise FaceFusionInputError("enabled must be bool")
    if not isinstance(cfg.mode, str) or not cfg.mode.strip():
        raise FaceFusionInputError("mode must be non-empty string")
    mode = cfg.mode.strip().lower()
    if mode not in FACE_FUSION_ALLOWED_MODES:
        raise FaceFusionInputError(f"invalid mode: {cfg.mode}")
    if not isinstance(cfg.reference_face_path, str) or not cfg.reference_face_path.strip():
        raise FaceFusionInputError("reference_face_path must be non-empty")
    if not isinstance(cfg.strength, (int, float)) or not (0.0 <= float(cfg.strength) <= 1.0):
        raise FaceFusionInputError("strength must be within [0,1]")
    if not isinstance(cfg.min_confidence, (int, float)) or not (
        0.0 <= float(cfg.min_confidence) <= 1.0
    ):
        raise FaceFusionInputError("min_confidence must be within [0,1]")
    if not isinstance(cfg.preserve_background, bool):
        raise FaceFusionInputError("preserve_background must be bool")


@dataclass(frozen=True)
class FaceFusionMetrics:
    stage_invocations: int = 0
    enabled_invocations: int = 0
    disabled_invocations: int = 0
    passthrough_invocations: int = 0
    has_face_invocations: int = 0
    no_face_invocations: int = 0
    error_count: int = 0


class FaceFusionObserver:
    def __init__(self) -> None:
        self.metrics = FaceFusionMetrics()

    def record_invocation(self, enabled: bool) -> None:
        self.metrics.stage_invocations += 1
        if enabled:
            self.metrics.enabled_invocations += 1
        else:
            self.metrics.disabled_invocations += 1

    def record_passthrough(self) -> None:
        self.metrics.passthrough_invocations += 1

    def record_face_presence(self, has_face: bool) -> None:
        if has_face:
            self.metrics.has_face_invocations += 1
        else:
            self.metrics.no_face_invocations += 1

    def record_error(self) -> None:
        self.metrics.error_count += 1


@dataclass(frozen=True)
class ROIBox:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if not all(isinstance(v, int) for v in (self.x, self.y, self.width, self.height)):
            raise TypeError("ROIBox coordinates must be integers")
        if self.x < 0 or self.y < 0:
            raise ValueError("ROIBox coordinates must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROIBox width and height must be positive")

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: dict) -> "ROIBox":
        return cls(
            x=int(data["x"]),
            y=int(data["y"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )


@dataclass(frozen=True)
class MappingEntry:
    source_range: Tuple[int, int]
    target_hue_range: Tuple[int, int]

    def __post_init__(self) -> None:
        _validate_tuple_range(self.source_range, "source_range")
        _validate_tuple_range(self.target_hue_range, "target_hue_range")

    def to_dict(self) -> dict:
        return {
            "source_range": list(self.source_range),
            "target_hue_range": list(self.target_hue_range),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MappingEntry":
        return cls(
            source_range=tuple(data["source_range"]),
            target_hue_range=tuple(data["target_hue_range"]),
        )


@dataclass(frozen=True)
class MappingConfig:
    version: str
    engine_mode: str
    mappings: Tuple[MappingEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not SEMVER_PATTERN.match(self.version):
            raise ValueError("version must follow semantic version format X.Y.Z")
        if self.engine_mode not in ENGINE_MODES:
            raise ValueError(f"engine_mode must be one of {ENGINE_MODES}")
        if not isinstance(self.mappings, tuple) or len(self.mappings) == 0:
            raise ValueError("mappings must be non-empty tuple")
        for entry in self.mappings:
            if not isinstance(entry, MappingEntry):
                raise TypeError("mappings must contain MappingEntry objects")

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "engine_mode": self.engine_mode,
            "mappings": [m.to_dict() for m in self.mappings],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MappingConfig":
        mappings = tuple(MappingEntry.from_dict(m) for m in data["mappings"])
        return cls(version=data["version"], engine_mode=data["engine_mode"], mappings=mappings)


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.width, int) or self.width <= 0:
            raise ValueError("width must be positive int")
        if not isinstance(self.height, int) or self.height <= 0:
            raise ValueError("height must be positive int")
        if not isinstance(self.fps, (int, float)) or self.fps <= 0:
            raise ValueError("fps must be positive number")
        if not isinstance(self.frame_count, int) or self.frame_count < 0:
            raise ValueError("frame_count must be non-negative int")

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VideoMetadata":
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            fps=float(data["fps"]),
            frame_count=int(data["frame_count"]),
        )


@dataclass(frozen=True)
class VideoResult:
    output_path: str
    subtitle_path: Optional[str]
    metadata: VideoMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.output_path, str) or not self.output_path:
            raise ValueError("output_path must be non-empty string")
        if self.subtitle_path is not None and not isinstance(self.subtitle_path, str):
            raise TypeError("subtitle_path must be string or None")
        if not isinstance(self.metadata, VideoMetadata):
            raise TypeError("metadata must be VideoMetadata")

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "subtitle_path": self.subtitle_path,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VideoResult":
        return cls(
            output_path=data["output_path"],
            subtitle_path=data.get("subtitle_path"),
            metadata=VideoMetadata.from_dict(data["metadata"]),
        )


@dataclass(frozen=True)
class VideoOutputContract:
    result: VideoResult

    def __post_init__(self) -> None:
        if not isinstance(self.result, VideoResult):
            raise TypeError("result must be VideoResult")

    def to_dict(self) -> dict:
        return {"result": self.result.to_dict()}
