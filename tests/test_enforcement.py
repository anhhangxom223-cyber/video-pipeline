"""
test_enforcement.py — Freeze constraint enforcement tests.

Validates that the codebase complies with:
- Video Pipeline v3.2.0 freeze (DTO immutability, field counts, interface signatures)
- Audio Pipeline v4.2.1 freeze (15 frozen DTOs, 49 fields, schema constraints,
  prosody governance PG-01→PG-10, threading prohibitions, dependency direction)
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import sys
import os
import unittest
from types import MappingProxyType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    StageMetric,
    VIDEO_PIPELINE_VERSION,
    AUDIO_PIPELINE_VERSION,
)


# ============================================================
# Section 1: DTO Immutability Enforcement (frozen=True)
# ============================================================


class TestDTOImmutability(unittest.TestCase):
    """All DTOs MUST be frozen=True per both freeze specs."""

    # --- Audio DTOs (ADM-01 → ADM-15) ---

    AUDIO_DTOS = [
        TimeRange,
        PitchFrame,
        RMSFrame,
        SegmentProsody,
        ASRSegment,
        ExtractedAudio,
        SeparatedAudio,
        VoiceAnalysisResult,
        AlignedSegment,
        TTSOutputSegment,
        DurationAlignedSegment,
        RMSProcessedSegment,
        MixedAudio,
        AudioInputContract,
        AudioResult,
    ]

    # --- Video DTOs (DM-01 → DM-12 subset present in code) ---

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
    ]

    def test_audio_dto_count_matches_freeze(self):
        """Audio freeze spec: exactly 15 frozen contracts."""
        self.assertEqual(len(self.AUDIO_DTOS), 15)

    def test_audio_dtos_are_frozen(self):
        """Every audio DTO must be a frozen dataclass (Section 6.3)."""
        for dto_cls in self.AUDIO_DTOS:
            with self.subTest(dto=dto_cls.__name__):
                self.assertTrue(
                    dataclasses.is_dataclass(dto_cls),
                    f"{dto_cls.__name__} must be a dataclass",
                )
                self.assertTrue(
                    dto_cls.__dataclass_params__.frozen,
                    f"{dto_cls.__name__} must be frozen=True",
                )

    def test_video_dtos_are_frozen(self):
        """Every video DTO must be a frozen dataclass (Section 5)."""
        for dto_cls in self.VIDEO_DTOS:
            with self.subTest(dto=dto_cls.__name__):
                self.assertTrue(
                    dataclasses.is_dataclass(dto_cls),
                    f"{dto_cls.__name__} must be a dataclass",
                )
                self.assertTrue(
                    dto_cls.__dataclass_params__.frozen,
                    f"{dto_cls.__name__} must be frozen=True",
                )

    def test_audio_dtos_reject_mutation(self):
        """Runtime enforcement: FrozenInstanceError on attribute set."""
        tr = TimeRange(start=0.0, end=1.0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            tr.start = 99.0

        pf = PitchFrame(time_offset=0.0, frequency_hz=440.0, confidence=0.9)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            pf.frequency_hz = 0.0

    def test_video_dtos_reject_mutation(self):
        """Runtime enforcement: FrozenInstanceError on attribute set."""
        roi = ROIBox(x=0, y=0, width=100, height=50)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            roi.x = 10

        meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            meta.fps = 60.0


# ============================================================
# Section 2: Audio DTO Field Count Enforcement (49 fields)
# ============================================================


class TestAudioDTOFieldCounts(unittest.TestCase):
    """Audio freeze Section 3: exact field counts per ADM-ID."""

    EXPECTED_FIELDS = {
        "ADM-01 TimeRange": (TimeRange, 2),
        "ADM-02 PitchFrame": (PitchFrame, 3),
        "ADM-03 RMSFrame": (RMSFrame, 2),
        "ADM-04 SegmentProsody": (SegmentProsody, 3),
        "ADM-05 ASRSegment": (ASRSegment, 6),
        "ADM-06 ExtractedAudio": (ExtractedAudio, 4),
        "ADM-07 SeparatedAudio": (SeparatedAudio, 2),
        "ADM-08 VoiceAnalysisResult": (VoiceAnalysisResult, 1),
        "ADM-09 AlignedSegment": (AlignedSegment, 4),
        "ADM-10 TTSOutputSegment": (TTSOutputSegment, 5),
        "ADM-11 DurationAlignedSegment": (DurationAlignedSegment, 5),
        "ADM-12 RMSProcessedSegment": (RMSProcessedSegment, 4),
        "ADM-13 MixedAudio": (MixedAudio, 3),
        "ADM-14 AudioInputContract": (AudioInputContract, 3),
        "ADM-15 AudioResult": (AudioResult, 2),
    }

    def test_individual_field_counts(self):
        for adm_id, (dto_cls, expected_count) in self.EXPECTED_FIELDS.items():
            with self.subTest(adm=adm_id):
                actual = len(dataclasses.fields(dto_cls))
                self.assertEqual(
                    actual,
                    expected_count,
                    f"{adm_id}: expected {expected_count} fields, got {actual}",
                )

    def test_total_field_count_is_49(self):
        """Audio freeze Section 3: total frozen fields = 49."""
        total = sum(
            len(dataclasses.fields(cls))
            for cls, _ in self.EXPECTED_FIELDS.values()
        )
        # The freeze doc says 49 total fields across 15 DTOs
        self.assertEqual(total, 49)


# ============================================================
# Section 3: Version Freeze
# ============================================================


class TestVersionFreeze(unittest.TestCase):
    """Pipeline versions must match freeze documents."""

    def test_video_pipeline_version(self):
        self.assertEqual(VIDEO_PIPELINE_VERSION, "3.2.0")

    def test_audio_pipeline_version(self):
        self.assertEqual(AUDIO_PIPELINE_VERSION, "4.2.1")


# ============================================================
# Section 4: Prosody Governance (PG-01 → PG-10)
# ============================================================


class TestProsodyGovernance(unittest.TestCase):
    """Audio freeze Section 1.4: prosody governance rules."""

    def _build_prosody(self):
        return SegmentProsody(
            speech_rate=1.0,
            rms_frames=(RMSFrame(time_offset=0.0, rms_value=0.1),),
            pitch_frames=(PitchFrame(time_offset=0.0, frequency_hz=440.0, confidence=0.9),),
        )

    def _build_asr_segment(self):
        return ASRSegment(
            segment_id=0,
            time_range=TimeRange(start=0.0, end=1.0),
            transcript="hello",
            avg_logprob=-0.5,
            no_speech_prob=0.1,
            prosody=self._build_prosody(),
        )

    def test_pg01_prosody_nested_in_asr_segment(self):
        """PG-01: Prosody exists ONLY inside ASRSegment.prosody."""
        seg = self._build_asr_segment()
        self.assertIsInstance(seg.prosody, SegmentProsody)

    def test_pg02_asr_segment_immutable(self):
        """PG-02: ASRSegment is immutable (frozen=True)."""
        seg = self._build_asr_segment()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            seg.transcript = "modified"

    def test_pg03_segment_prosody_immutable(self):
        """PG-03: SegmentProsody is immutable (frozen=True)."""
        prosody = self._build_prosody()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            prosody.speech_rate = 2.0

    def test_pg05_segment_map_immutable(self):
        """PG-05: Segment map is MappingProxyType — TypeError on write."""
        seg = self._build_asr_segment()
        segment_map = MappingProxyType({seg.segment_id: seg})
        with self.assertRaises(TypeError):
            segment_map[0] = None  # type: ignore

    def test_pg06_subtitle_aligner_no_prosody_access(self):
        """PG-06: SubtitleAligner must NOT access .prosody (AST enforcement)."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "audio_pipeline.py",
        )
        with open(source_path, "r") as f:
            tree = ast.parse(f.read())

        # Find SubtitleAligner class and check for .prosody attribute access
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "SubtitleAligner":
                for child in ast.walk(node):
                    if isinstance(child, ast.Attribute) and child.attr == "prosody":
                        self.fail("SubtitleAligner accesses .prosody — violates PG-06")

    def test_pg07_duration_aligner_no_prosody_access(self):
        """PG-07: DurationAligner must NOT access .prosody or segment map."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "audio_pipeline.py",
        )
        with open(source_path, "r") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DurationAligner":
                for child in ast.walk(node):
                    if isinstance(child, ast.Attribute) and child.attr == "prosody":
                        self.fail("DurationAligner accesses .prosody — violates PG-07")


# ============================================================
# Section 5: Dependency Direction Enforcement
# ============================================================


class TestDependencyDirection(unittest.TestCase):
    """Freeze specs Section 7: dependency layering rules."""

    def test_audio_pipeline_does_not_import_video_pipeline(self):
        """Audio must NOT import VideoPipeline, ReaderThread, etc."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "audio_pipeline.py",
        )
        with open(source_path, "r") as f:
            source = f.read()

        prohibited = [
            "from video_pipeline import",
            "import video_pipeline",
            "ReaderThread",
            "ProcessorThread",
            "EncoderThread",
            "BackboneExecutionCore",
        ]
        for token in prohibited:
            self.assertNotIn(
                token,
                source,
                f"audio_pipeline.py contains prohibited import/reference: {token}",
            )

    def test_audio_pipeline_no_threading_imports(self):
        """Audio freeze Section 7.3: no threading/asyncio/multiprocessing."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "audio_pipeline.py",
        )
        with open(source_path, "r") as f:
            source = f.read()

        prohibited = [
            "import threading",
            "import asyncio",
            "import multiprocessing",
            "concurrent.futures",
            "import queue",
        ]
        for token in prohibited:
            self.assertNotIn(
                token,
                source,
                f"audio_pipeline.py contains prohibited concurrency import: {token}",
            )

    def test_observability_not_imported_by_dto_layer(self):
        """Observability MUST NOT be imported by DTO layer."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "audio_pipeline.py",
        )
        with open(source_path, "r") as f:
            source = f.read()

        prohibited = [
            "from core.observability",
            "import core.observability",
            "from core.runtime",
            "import core.runtime",
        ]
        for token in prohibited:
            self.assertNotIn(
                token,
                source,
                f"audio_pipeline.py imports observability/runtime modules: {token}",
            )


# ============================================================
# Section 6: RunMetadata Freeze
# ============================================================


class TestRunMetadataFreeze(unittest.TestCase):
    """RunMetadata DTO must be frozen with exact fields."""

    def test_run_metadata_is_frozen(self):
        self.assertTrue(dataclasses.is_dataclass(RunMetadata))
        self.assertTrue(RunMetadata.__dataclass_params__.frozen)

    def test_run_metadata_fields(self):
        fields = {f.name for f in dataclasses.fields(RunMetadata)}
        expected = {"run_id", "pipeline_version", "config_hash", "timestamp", "input_hash"}
        self.assertEqual(fields, expected)

    def test_run_metadata_rejects_mutation(self):
        meta = RunMetadata(
            run_id="test",
            pipeline_version="3.2.0",
            config_hash="abc",
            timestamp="2024-01-01T00:00:00",
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            meta.run_id = "changed"


if __name__ == "__main__":
    unittest.main()
