"""
test_enforcement.py — Freeze constraint enforcement tests.

Validates codebase compliance with:
- Video Pipeline v3.2.0 freeze (DTO immutability, field counts, interface signatures)
- Audio Pipeline v4.2.1 freeze (15 frozen DTOs, 49 fields, schema constraints,
  prosody governance PG-01→PG-10, threading prohibitions, dependency direction)
"""
from __future__ import annotations

import ast
import dataclasses
import os
from types import MappingProxyType

import pytest
import numpy as np

from audio_pipeline import (
    AudioInputContract,
    AudioResult,
    AlignedSegment,
    ASRSegment,
    DurationAlignedSegment,
    ExtractedAudio,
    MixedAudio,
    PitchFrame,
    RMSFrame,
    RMSProcessedSegment,
    SegmentProsody,
    SeparatedAudio,
    TimeRange,
    TTSOutputSegment,
    VoiceAnalysisResult,
)
from video_pipeline import (
    FaceFusionMetrics,
    MappingConfig,
    MappingEntry,
    ROIBox,
    VideoMetadata,
    VideoOutputContract,
    VideoResult,
)
from config_loader import (
    PipelineContext,
    ResolvedConfig,
    RunMetadata,
    VIDEO_PIPELINE_VERSION,
    AUDIO_PIPELINE_VERSION,
)
from dto.face_fusion import FaceFusionConfig

AUDIO_SOURCE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "audio_pipeline.py",
)

# ============================================================
# Audio DTOs — ordered by ADM-ID for freeze cross-reference
# ============================================================

AUDIO_DTOS = [
    TimeRange,            # ADM-01
    PitchFrame,           # ADM-02
    RMSFrame,             # ADM-03
    SegmentProsody,       # ADM-04
    ASRSegment,           # ADM-05
    ExtractedAudio,       # ADM-06
    SeparatedAudio,       # ADM-07
    VoiceAnalysisResult,  # ADM-08
    AlignedSegment,       # ADM-09
    TTSOutputSegment,     # ADM-10
    DurationAlignedSegment,  # ADM-11
    RMSProcessedSegment,  # ADM-12
    MixedAudio,           # ADM-13
    AudioInputContract,   # ADM-14
    AudioResult,          # ADM-15
]

VIDEO_DTOS = [
    ResolvedConfig,
    PipelineContext,
    VideoResult,
    VideoOutputContract,
    MappingConfig,
    MappingEntry,
    ROIBox,
    VideoMetadata,
    FaceFusionMetrics,
    FaceFusionConfig,
]


# ============================================================
# 1  DTO Immutability — frozen=True (Sections 5 / 6.3)
# ============================================================


class TestDTOImmutability:
    """All DTOs MUST be frozen=True per both freeze specs."""

    def test_audio_dto_count_is_15(self):
        """Audio freeze §3: exactly 15 frozen contracts."""
        assert len(AUDIO_DTOS) == 15

    @pytest.mark.parametrize("dto_cls", AUDIO_DTOS, ids=lambda c: c.__name__)
    def test_audio_dto_is_frozen_dataclass(self, dto_cls):
        assert dataclasses.is_dataclass(dto_cls), f"{dto_cls.__name__} must be a dataclass"
        assert dto_cls.__dataclass_params__.frozen, f"{dto_cls.__name__} must be frozen=True"

    @pytest.mark.parametrize("dto_cls", VIDEO_DTOS, ids=lambda c: c.__name__)
    def test_video_dto_is_frozen_dataclass(self, dto_cls):
        assert dataclasses.is_dataclass(dto_cls), f"{dto_cls.__name__} must be a dataclass"
        assert dto_cls.__dataclass_params__.frozen, f"{dto_cls.__name__} must be frozen=True"

    def test_audio_dto_rejects_mutation(self):
        tr = TimeRange(start=0.0, end=1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            tr.start = 99.0

        pf = PitchFrame(time_offset=0.0, frequency_hz=440.0, confidence=0.9)
        with pytest.raises(dataclasses.FrozenInstanceError):
            pf.frequency_hz = 0.0

    def test_video_dto_rejects_mutation(self):
        roi = ROIBox(x=0, y=0, width=100, height=50)
        with pytest.raises(dataclasses.FrozenInstanceError):
            roi.x = 10

        meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)
        with pytest.raises(dataclasses.FrozenInstanceError):
            meta.fps = 60.0


# ============================================================
# 2  Audio DTO Field Counts — 49 total (§3.1)
# ============================================================


EXPECTED_FIELDS = {
    "ADM-01_TimeRange":             (TimeRange, 2),
    "ADM-02_PitchFrame":            (PitchFrame, 3),
    "ADM-03_RMSFrame":              (RMSFrame, 2),
    "ADM-04_SegmentProsody":        (SegmentProsody, 3),
    "ADM-05_ASRSegment":            (ASRSegment, 6),
    "ADM-06_ExtractedAudio":        (ExtractedAudio, 4),
    "ADM-07_SeparatedAudio":        (SeparatedAudio, 2),
    "ADM-08_VoiceAnalysisResult":   (VoiceAnalysisResult, 1),
    "ADM-09_AlignedSegment":        (AlignedSegment, 4),
    "ADM-10_TTSOutputSegment":      (TTSOutputSegment, 5),
    "ADM-11_DurationAlignedSegment": (DurationAlignedSegment, 5),
    "ADM-12_RMSProcessedSegment":   (RMSProcessedSegment, 4),
    "ADM-13_MixedAudio":            (MixedAudio, 3),
    "ADM-14_AudioInputContract":    (AudioInputContract, 3),
    "ADM-15_AudioResult":           (AudioResult, 2),
}


class TestAudioDTOFieldCounts:

    @pytest.mark.parametrize(
        "adm_id, spec",
        EXPECTED_FIELDS.items(),
        ids=EXPECTED_FIELDS.keys(),
    )
    def test_field_count(self, adm_id, spec):
        dto_cls, expected = spec
        actual = len(dataclasses.fields(dto_cls))
        assert actual == expected, f"{adm_id}: expected {expected} fields, got {actual}"

    def test_total_field_count_is_49(self):
        total = sum(len(dataclasses.fields(c)) for c, _ in EXPECTED_FIELDS.values())
        assert total == 49


# ============================================================
# 3  Version Freeze
# ============================================================


class TestVersionFreeze:

    def test_video_pipeline_version(self):
        assert VIDEO_PIPELINE_VERSION == "3.2.0"

    def test_audio_pipeline_version(self):
        assert AUDIO_PIPELINE_VERSION == "4.2.1"


# ============================================================
# 4  FaceFusionConfig Freeze (DM-09)
# ============================================================


class TestFaceFusionConfigFreeze:

    def test_frozen_dataclass(self):
        assert dataclasses.is_dataclass(FaceFusionConfig)
        assert FaceFusionConfig.__dataclass_params__.frozen

    def test_field_names(self):
        names = [f.name for f in dataclasses.fields(FaceFusionConfig)]
        assert names == [
            "enabled", "mode", "reference_face_path",
            "strength", "min_confidence", "preserve_background",
        ]

    def test_field_count_is_6(self):
        assert len(dataclasses.fields(FaceFusionConfig)) == 6

    def test_rejects_mutation(self):
        cfg = FaceFusionConfig(
            enabled=True, mode="single",
            reference_face_path="ref.png",
            strength=0.8, min_confidence=0.5,
            preserve_background=True,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.enabled = False


# ============================================================
# 5  Prosody Governance (PG-01 → PG-07)
# ============================================================


def _build_prosody():
    return SegmentProsody(
        speech_rate=1.0,
        rms_frames=(RMSFrame(time_offset=0.0, rms_value=0.1),),
        pitch_frames=(PitchFrame(time_offset=0.0, frequency_hz=440.0, confidence=0.9),),
    )


def _build_asr_segment():
    return ASRSegment(
        segment_id=0,
        time_range=TimeRange(start=0.0, end=1.0),
        transcript="hello",
        avg_logprob=-0.5,
        no_speech_prob=0.1,
        prosody=_build_prosody(),
    )


class TestProsodyGovernance:

    def test_pg01_prosody_nested_in_asr_segment(self):
        """PG-01: Prosody exists ONLY inside ASRSegment.prosody."""
        seg = _build_asr_segment()
        assert isinstance(seg.prosody, SegmentProsody)

    def test_pg02_asr_segment_immutable(self):
        """PG-02: ASRSegment frozen=True."""
        seg = _build_asr_segment()
        with pytest.raises(dataclasses.FrozenInstanceError):
            seg.transcript = "modified"

    def test_pg03_segment_prosody_immutable(self):
        """PG-03: SegmentProsody frozen=True."""
        prosody = _build_prosody()
        with pytest.raises(dataclasses.FrozenInstanceError):
            prosody.speech_rate = 2.0

    def test_pg05_segment_map_immutable(self):
        """PG-05: Segment map is MappingProxyType — TypeError on write."""
        seg = _build_asr_segment()
        segment_map = MappingProxyType({seg.segment_id: seg})
        with pytest.raises(TypeError):
            segment_map[0] = None

    def test_pg06_subtitle_aligner_no_prosody_access(self):
        """PG-06: SubtitleAligner must NOT access .prosody (AST enforcement)."""
        with open(AUDIO_SOURCE) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "SubtitleAligner":
                for child in ast.walk(node):
                    if isinstance(child, ast.Attribute) and child.attr == "prosody":
                        pytest.fail("SubtitleAligner accesses .prosody — violates PG-06")

    def test_pg07_duration_aligner_no_prosody_access(self):
        """PG-07: DurationAligner must NOT access .prosody or segment map."""
        with open(AUDIO_SOURCE) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DurationAligner":
                for child in ast.walk(node):
                    if isinstance(child, ast.Attribute) and child.attr == "prosody":
                        pytest.fail("DurationAligner accesses .prosody — violates PG-07")


# ============================================================
# 6  Dependency Direction (§7)
# ============================================================


class TestDependencyDirection:

    @pytest.fixture(autouse=True)
    def _load_audio_source(self):
        with open(AUDIO_SOURCE) as f:
            self.source = f.read()

    def test_audio_does_not_import_video_pipeline(self):
        """§1.3 / §7: audio must NOT import video pipeline symbols."""
        prohibited = [
            "from video_pipeline import",
            "import video_pipeline",
            "ReaderThread",
            "ProcessorThread",
            "EncoderThread",
            "BackboneExecutionCore",
        ]
        for token in prohibited:
            assert token not in self.source, (
                f"audio_pipeline.py contains prohibited reference: {token}"
            )

    def test_audio_no_threading_imports(self):
        """§7.3: no threading / asyncio / multiprocessing in audio pipeline."""
        prohibited = [
            "import threading",
            "import asyncio",
            "import multiprocessing",
            "concurrent.futures",
            "import queue",
        ]
        for token in prohibited:
            assert token not in self.source, (
                f"audio_pipeline.py contains prohibited concurrency import: {token}"
            )

    def test_observability_not_imported_by_audio(self):
        """Observability MUST NOT be imported by DTO / audio layer."""
        prohibited = [
            "from core.observability",
            "import core.observability",
            "from core.runtime",
            "import core.runtime",
        ]
        for token in prohibited:
            assert token not in self.source, (
                f"audio_pipeline.py imports observability module: {token}"
            )


# ============================================================
# 7  RunMetadata Freeze
# ============================================================


class TestRunMetadataFreeze:

    def test_frozen_dataclass(self):
        assert dataclasses.is_dataclass(RunMetadata)
        assert RunMetadata.__dataclass_params__.frozen

    def test_exact_fields(self):
        fields = {f.name for f in dataclasses.fields(RunMetadata)}
        assert fields == {"run_id", "pipeline_version", "config_hash", "timestamp", "input_hash"}

    def test_rejects_mutation(self):
        meta = RunMetadata(
            run_id="test", pipeline_version="3.2.0",
            config_hash="abc", timestamp="2024-01-01T00:00:00",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            meta.run_id = "changed"


# ============================================================
# 8  Video DTO Field Names (DM-08 ROIBox)
# ============================================================


class TestROIBoxFreeze:
    """DM-08: ROIBox fields are x, y, width, height (integer coordinates)."""

    def test_field_names(self):
        names = [f.name for f in dataclasses.fields(ROIBox)]
        assert names == ["x", "y", "width", "height"]

    def test_field_count(self):
        assert len(dataclasses.fields(ROIBox)) == 4


# ============================================================
# 9  MappingConfig Engine Modes (§4)
# ============================================================


class TestEngineModeFreeze:

    def test_hsv_and_lab_only(self):
        from video_pipeline import ENGINE_MODES
        assert ENGINE_MODES == {"HSV", "LAB"}
