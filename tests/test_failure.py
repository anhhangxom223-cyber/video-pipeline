"""
test_failure.py — Failure path and error handling tests.

Tests invalid inputs, error hierarchies, boundary violations, malformed data,
checkpoint corruption, and debug report generation under failure conditions.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from audio_pipeline import (
    AudioInputContract,
    AudioPipelineError,
    AudioExtractionError,
    AudioMixingError,
    DurationAlignmentError,
    ProsodyAccessViolation,
    RMSProcessingError,
    SegmentMapError,
    SourceSeparationError,
    SubtitleAlignmentError,
    TTSGenerationError,
    VoiceAnalysisError,
    AudioResult,
    TimeRange,
    PitchFrame,
    RMSFrame,
    SegmentProsody,
    ASRSegment,
    ExtractedAudio,
    _read_wav,
)
from video_pipeline import (
    MappingConfig,
    MappingEntry,
    ROIBox,
    VideoMetadata,
    VideoResult,
)
from config_loader import (
    ConfigLoader,
    PipelineMetrics,
    ResolvedConfig,
    RunMetadata,
    StageCheckpoint,
)
from core.runtime.run_folder_manager import RunFolderManager
from core.runtime.checkpoint_runner import (
    CheckpointRunner,
    CheckpointSerializationError,
    CheckpointDeserializationError,
)
from core.observability.failure_middleware import FailureCaptureMiddleware
from core.observability.debug_report import DebugReportGenerator


# ============================================================
# Section 1: Audio Error Hierarchy
# ============================================================


class TestAudioErrorHierarchy(unittest.TestCase):
    """Audio pipeline defines a strict error hierarchy rooted at AudioPipelineError."""

    SUBCLASSES = [
        AudioExtractionError,
        SourceSeparationError,
        VoiceAnalysisError,
        SubtitleAlignmentError,
        TTSGenerationError,
        DurationAlignmentError,
        RMSProcessingError,
        AudioMixingError,
        ProsodyAccessViolation,
        SegmentMapError,
    ]

    def test_all_inherit_from_audio_pipeline_error(self):
        for cls in self.SUBCLASSES:
            with self.subTest(error=cls.__name__):
                self.assertTrue(
                    issubclass(cls, AudioPipelineError),
                    f"{cls.__name__} must inherit from AudioPipelineError",
                )

    def test_all_inherit_from_exception(self):
        for cls in self.SUBCLASSES:
            with self.subTest(error=cls.__name__):
                self.assertTrue(issubclass(cls, Exception))

    def test_raise_and_catch_by_base(self):
        """Catching AudioPipelineError catches any stage-specific error."""
        for cls in self.SUBCLASSES:
            with self.subTest(error=cls.__name__):
                with self.assertRaises(AudioPipelineError):
                    raise cls(f"test error from {cls.__name__}")

    def test_error_message_preserved(self):
        err = AudioExtractionError("extraction failed: bad format")
        self.assertIn("extraction failed", str(err))


# ============================================================
# Section 2: Invalid DTO Construction
# ============================================================


class TestInvalidDTOConstruction(unittest.TestCase):
    """DTOs with post-init validation must reject invalid inputs."""

    # --- ROIBox ---

    def test_roi_box_non_integer_coordinates(self):
        with self.assertRaises(TypeError):
            ROIBox(x=1.5, y=0, width=100, height=50)

    def test_roi_box_negative_x(self):
        with self.assertRaises(ValueError):
            ROIBox(x=-1, y=0, width=100, height=50)

    def test_roi_box_zero_width(self):
        with self.assertRaises(ValueError):
            ROIBox(x=0, y=0, width=0, height=50)

    def test_roi_box_negative_height(self):
        with self.assertRaises(ValueError):
            ROIBox(x=0, y=0, width=10, height=-5)

    # --- MappingEntry ---

    def test_mapping_entry_reversed_range(self):
        with self.assertRaises(ValueError):
            MappingEntry(source_range=(100, 10), target_hue_range=(0, 50))

    def test_mapping_entry_out_of_range(self):
        with self.assertRaises(ValueError):
            MappingEntry(source_range=(0, 300), target_hue_range=(0, 50))

    def test_mapping_entry_non_tuple(self):
        with self.assertRaises(TypeError):
            MappingEntry(source_range=[0, 50], target_hue_range=(0, 50))

    # --- MappingConfig ---

    def test_mapping_config_empty_mappings(self):
        with self.assertRaises(ValueError):
            MappingConfig(version="1.0.0", engine_mode="HSV", mappings=())

    def test_mapping_config_invalid_engine_mode(self):
        with self.assertRaises(ValueError):
            MappingConfig(
                version="1.0.0",
                engine_mode="INVALID",
                mappings=(MappingEntry(source_range=(0, 50), target_hue_range=(100, 150)),),
            )

    def test_mapping_config_bad_version_format(self):
        with self.assertRaises(ValueError):
            MappingConfig(
                version="v1",
                engine_mode="HSV",
                mappings=(MappingEntry(source_range=(0, 50), target_hue_range=(100, 150)),),
            )

    # --- VideoMetadata ---

    def test_video_metadata_zero_width(self):
        with self.assertRaises(ValueError):
            VideoMetadata(width=0, height=240, fps=30.0, frame_count=10)

    def test_video_metadata_negative_fps(self):
        with self.assertRaises(ValueError):
            VideoMetadata(width=320, height=240, fps=-1.0, frame_count=10)

    def test_video_metadata_negative_frame_count(self):
        with self.assertRaises(ValueError):
            VideoMetadata(width=320, height=240, fps=30.0, frame_count=-1)

    # --- VideoResult ---

    def test_video_result_empty_output_path(self):
        meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)
        with self.assertRaises(ValueError):
            VideoResult(output_path="", subtitle_path=None, metadata=meta)

    def test_video_result_wrong_metadata_type(self):
        with self.assertRaises(TypeError):
            VideoResult(output_path="/tmp/out.mp4", subtitle_path=None, metadata={"not": "VideoMetadata"})

    # --- ResolvedConfig ---

    def test_resolved_config_non_dict(self):
        with self.assertRaises(TypeError):
            ResolvedConfig(configuration_parameters="not a dict")

    def test_resolved_config_non_string_values(self):
        with self.assertRaises(TypeError):
            ResolvedConfig(configuration_parameters={"key": 123})


# ============================================================
# Section 3: ConfigLoader Validation Failures
# ============================================================


class TestConfigLoaderFailures(unittest.TestCase):
    """ConfigLoader must reject invalid configurations."""

    def test_invalid_face_fusion_mode(self):
        with self.assertRaises(Exception):
            ConfigLoader.resolve(cli_args={"face_fusion_mode": "invalid_mode"})

    def test_face_fusion_strength_out_of_range(self):
        with self.assertRaises(ValueError):
            ConfigLoader.resolve(cli_args={"face_fusion_strength": "2.0"})

    def test_face_fusion_min_confidence_out_of_range(self):
        with self.assertRaises(ValueError):
            ConfigLoader.resolve(cli_args={"face_fusion_min_confidence": "-0.1"})

    def test_face_fusion_enabled_invalid_bool(self):
        with self.assertRaises(ValueError):
            ConfigLoader.resolve(cli_args={"face_fusion_enabled": "maybe"})

    def test_empty_config_key_rejected(self):
        with self.assertRaises(ValueError):
            ConfigLoader.resolve(cli_args={"": "value"})

    def test_non_dict_json_rejected(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump([1, 2, 3], f)
            path = f.name

        try:
            with self.assertRaises(TypeError):
                ConfigLoader.load_json(path)
        finally:
            os.unlink(path)


# ============================================================
# Section 4: Checkpoint Corruption & Deserialization Failures
# ============================================================


class TestCheckpointFailures(unittest.TestCase):
    """Checkpoint runner must handle corrupt and malformed data."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.checkpoint = StageCheckpoint(self.tmpdir)
        self.metrics = PipelineMetrics("test-run")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_deserialization_unknown_type(self):
        """Unknown __type__ in envelope returns raw data dict."""
        runner = CheckpointRunner(
            self.checkpoint, self.metrics, type_registry={"VideoResult": VideoResult}
        )
        self.checkpoint.save("TestStage", {
            "__type__": "UnknownType",
            "data": {"key": "value"},
        })

        result = runner.run_stage("TestStage", lambda: None)
        # Falls back to raw data dict
        self.assertIsInstance(result, dict)
        self.assertEqual(result["key"], "value")

    def test_deserialization_no_envelope(self):
        """Legacy checkpoint without __type__ returns raw dict."""
        runner = CheckpointRunner(self.checkpoint, self.metrics)
        self.checkpoint.save("TestStage", {"legacy": "data"})

        result = runner.run_stage("TestStage", lambda: None)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["legacy"], "data")

    def test_deserialization_corrupt_data(self):
        """Malformed data inside a typed envelope raises CheckpointDeserializationError."""
        runner = CheckpointRunner(
            self.checkpoint, self.metrics,
            type_registry={"VideoResult": VideoResult},
        )
        self.checkpoint.save("TestStage", {
            "__type__": "VideoResult",
            "data": {"bad_field": "no output_path"},
        })

        with self.assertRaises(CheckpointDeserializationError):
            runner.run_stage("TestStage", lambda: None)

    def test_serialization_rejects_non_serializable(self):
        """Stage result without to_dict() raises CheckpointSerializationError."""
        runner = CheckpointRunner(self.checkpoint, self.metrics)

        with self.assertRaises(CheckpointSerializationError):
            runner.run_stage("BadStage", lambda: 42)


# ============================================================
# Section 5: WAV Read Failures
# ============================================================


class TestWavReadFailures(unittest.TestCase):

    def test_read_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            _read_wav("/nonexistent/audio.wav")

    def test_read_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            with self.assertRaises(Exception):
                _read_wav(path)
        finally:
            os.unlink(path)


# ============================================================
# Section 6: Debug Report Generation Under Failure
# ============================================================


class TestDebugReportGeneration(unittest.TestCase):
    """DebugReportGenerator produces valid JSON on pipeline failure."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_generates_report_with_all_sections(self):
        config = ResolvedConfig.from_dict({"input_path": "/tmp/input.mp4"})
        metrics = PipelineMetrics("test-run")
        metric = metrics.begin_stage("FailingStage")
        metrics.end_stage(metric, "failed", "something broke")

        run_metadata = RunMetadata(
            run_id="test-run",
            pipeline_version="3.2.0",
            config_hash="abc123",
            timestamp="2024-01-01T00:00:00",
        )

        output_path = os.path.join(self.tmpdir, "debug_report.json")

        try:
            raise RuntimeError("test failure")
        except RuntimeError as e:
            DebugReportGenerator.generate(
                run_metadata=run_metadata,
                config=config,
                metrics=metrics,
                error=e,
                checkpoint_dir=self.tmpdir,
                output_path=output_path,
            )

        self.assertTrue(os.path.isfile(output_path))

        with open(output_path) as f:
            report = json.load(f)

        # Verify all required sections
        self.assertIn("pipeline", report)
        self.assertIn("error", report)
        self.assertIn("config", report)
        self.assertIn("environment", report)

        # Verify error details
        self.assertEqual(report["error"]["type"], "RuntimeError")
        self.assertIn("test failure", report["error"]["message"])

        # Verify failed stage detection
        self.assertEqual(report["pipeline"]["failed_stage"], "FailingStage")
        self.assertEqual(report["pipeline"]["run_id"], "test-run")

    def test_report_with_no_metrics(self):
        config = ResolvedConfig.from_dict({})
        run_metadata = RunMetadata(
            run_id="x", pipeline_version="3.2.0",
            config_hash="", timestamp="",
        )
        output_path = os.path.join(self.tmpdir, "report.json")

        try:
            raise ValueError("boom")
        except ValueError as e:
            DebugReportGenerator.generate(
                run_metadata=run_metadata,
                config=config,
                metrics=None,
                error=e,
                checkpoint_dir=None,
                output_path=output_path,
            )

        with open(output_path) as f:
            report = json.load(f)

        self.assertIsNone(report["pipeline"]["failed_stage"])
        self.assertIsNone(report["pipeline"]["metrics"])


# ============================================================
# Section 7: FailureCaptureMiddleware Failure Path
# ============================================================


class TestFailureCaptureMiddlewareErrors(unittest.TestCase):
    """Middleware must re-raise after generating debug report."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.run_folder = RunFolderManager(base_dir=self.tmpdir)
        self.run_folder.create("error_test")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_reraises_original_exception(self):
        config = ResolvedConfig.from_dict({})
        run_metadata = RunMetadata(
            run_id="err", pipeline_version="3.2.0",
            config_hash="", timestamp="",
        )
        metrics = PipelineMetrics("err")
        middleware = FailureCaptureMiddleware(
            run_metadata=run_metadata,
            config=config,
            metrics=metrics,
            run_folder=self.run_folder,
        )

        with self.assertRaises(AudioExtractionError) as ctx:
            middleware.run(lambda: (_ for _ in ()).throw(AudioExtractionError("bad audio")))

        self.assertIn("bad audio", str(ctx.exception))

    def test_debug_report_created_on_audio_error(self):
        config = ResolvedConfig.from_dict({})
        run_metadata = RunMetadata(
            run_id="err", pipeline_version="3.2.0",
            config_hash="", timestamp="",
        )
        metrics = PipelineMetrics("err")
        middleware = FailureCaptureMiddleware(
            run_metadata=run_metadata,
            config=config,
            metrics=metrics,
            run_folder=self.run_folder,
        )

        def fail():
            raise SourceSeparationError("separation failed")

        with self.assertRaises(SourceSeparationError):
            middleware.run(fail)

        self.assertTrue(os.path.isfile(self.run_folder.debug_report_path()))


# ============================================================
# Section 8: Metrics Failure Tracking
# ============================================================


class TestMetricsFailureTracking(unittest.TestCase):

    def test_failed_stage_recorded(self):
        metrics = PipelineMetrics("test-run")
        m = metrics.begin_stage("Stage1")
        metrics.end_stage(m, "failed", "something broke")

        summary = metrics.get_summary()
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["passed"], 0)
        self.assertEqual(summary["stages"][0]["status"], "failed")
        self.assertEqual(summary["stages"][0]["error"], "something broke")

    def test_mixed_success_and_failure(self):
        metrics = PipelineMetrics("test-run")

        m1 = metrics.begin_stage("Stage1")
        metrics.end_stage(m1, "success")

        m2 = metrics.begin_stage("Stage2")
        metrics.end_stage(m2, "failed", "error")

        summary = metrics.get_summary()
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["total_stages"], 2)


if __name__ == "__main__":
    unittest.main()
