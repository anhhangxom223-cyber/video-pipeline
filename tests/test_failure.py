"""
test_failure.py — Failure path and error handling tests.

Tests invalid inputs, error hierarchies, boundary violations, malformed data,
checkpoint corruption, and debug report generation under failure conditions.
"""
from __future__ import annotations

import dataclasses
import json
import os

import pytest
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
# 1  Audio Error Hierarchy
# ============================================================


AUDIO_ERROR_SUBCLASSES = [
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


class TestAudioErrorHierarchy:

    @pytest.mark.parametrize("cls", AUDIO_ERROR_SUBCLASSES, ids=lambda c: c.__name__)
    def test_inherits_from_audio_pipeline_error(self, cls):
        assert issubclass(cls, AudioPipelineError)

    @pytest.mark.parametrize("cls", AUDIO_ERROR_SUBCLASSES, ids=lambda c: c.__name__)
    def test_inherits_from_exception(self, cls):
        assert issubclass(cls, Exception)

    @pytest.mark.parametrize("cls", AUDIO_ERROR_SUBCLASSES, ids=lambda c: c.__name__)
    def test_catchable_by_base(self, cls):
        with pytest.raises(AudioPipelineError):
            raise cls(f"test error from {cls.__name__}")

    def test_error_message_preserved(self):
        err = AudioExtractionError("extraction failed: bad format")
        assert "extraction failed" in str(err)


# ============================================================
# 2  Invalid DTO Construction
# ============================================================


class TestInvalidROIBox:

    def test_non_integer_coordinates(self):
        with pytest.raises(TypeError):
            ROIBox(x=1.5, y=0, width=100, height=50)

    def test_negative_x(self):
        with pytest.raises(ValueError):
            ROIBox(x=-1, y=0, width=100, height=50)

    def test_zero_width(self):
        with pytest.raises(ValueError):
            ROIBox(x=0, y=0, width=0, height=50)

    def test_negative_height(self):
        with pytest.raises(ValueError):
            ROIBox(x=0, y=0, width=10, height=-5)


class TestInvalidMappingEntry:

    def test_reversed_range(self):
        with pytest.raises(ValueError):
            MappingEntry(source_range=(100, 10), target_hue_range=(0, 50))

    def test_out_of_range(self):
        with pytest.raises(ValueError):
            MappingEntry(source_range=(0, 300), target_hue_range=(0, 50))

    def test_non_tuple(self):
        with pytest.raises(TypeError):
            MappingEntry(source_range=[0, 50], target_hue_range=(0, 50))


class TestInvalidMappingConfig:

    def test_empty_mappings(self):
        with pytest.raises(ValueError):
            MappingConfig(version="1.0.0", engine_mode="HSV", mappings=())

    def test_invalid_engine_mode(self):
        with pytest.raises(ValueError):
            MappingConfig(
                version="1.0.0", engine_mode="INVALID",
                mappings=(MappingEntry(source_range=(0, 50), target_hue_range=(100, 150)),),
            )

    def test_bad_version_format(self):
        with pytest.raises(ValueError):
            MappingConfig(
                version="v1", engine_mode="HSV",
                mappings=(MappingEntry(source_range=(0, 50), target_hue_range=(100, 150)),),
            )


class TestInvalidVideoMetadata:

    def test_zero_width(self):
        with pytest.raises(ValueError):
            VideoMetadata(width=0, height=240, fps=30.0, frame_count=10)

    def test_negative_fps(self):
        with pytest.raises(ValueError):
            VideoMetadata(width=320, height=240, fps=-1.0, frame_count=10)

    def test_negative_frame_count(self):
        with pytest.raises(ValueError):
            VideoMetadata(width=320, height=240, fps=30.0, frame_count=-1)


class TestInvalidVideoResult:

    def test_empty_output_path(self):
        meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)
        with pytest.raises(ValueError):
            VideoResult(output_path="", subtitle_path=None, metadata=meta)

    def test_wrong_metadata_type(self):
        with pytest.raises(TypeError):
            VideoResult(output_path="/tmp/out.mp4", subtitle_path=None, metadata={"not": "VideoMetadata"})


class TestInvalidResolvedConfig:

    def test_non_dict(self):
        with pytest.raises(TypeError):
            ResolvedConfig(configuration_parameters="not a dict")

    def test_non_string_values(self):
        with pytest.raises(TypeError):
            ResolvedConfig(configuration_parameters={"key": 123})


# ============================================================
# 3  ConfigLoader Validation Failures
# ============================================================


class TestConfigLoaderFailures:

    def test_invalid_face_fusion_mode(self):
        with pytest.raises(Exception):
            ConfigLoader.resolve(cli_args={"face_fusion_mode": "invalid_mode"})

    def test_strength_out_of_range(self):
        with pytest.raises(ValueError):
            ConfigLoader.resolve(cli_args={"face_fusion_strength": "2.0"})

    def test_min_confidence_out_of_range(self):
        with pytest.raises(ValueError):
            ConfigLoader.resolve(cli_args={"face_fusion_min_confidence": "-0.1"})

    def test_enabled_invalid_bool(self):
        with pytest.raises(ValueError):
            ConfigLoader.resolve(cli_args={"face_fusion_enabled": "maybe"})

    def test_empty_config_key_rejected(self):
        with pytest.raises(ValueError):
            ConfigLoader.resolve(cli_args={"": "value"})

    def test_non_dict_json_rejected(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(TypeError):
            ConfigLoader.load_json(str(p))


# ============================================================
# 4  Checkpoint Corruption & Deserialization
# ============================================================


class TestCheckpointFailures:

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.checkpoint = StageCheckpoint(str(tmp_path))
        self.metrics = PipelineMetrics("test-run")

    def test_unknown_type_returns_raw_data(self):
        runner = CheckpointRunner(
            self.checkpoint, self.metrics,
            type_registry={"VideoResult": VideoResult},
        )
        self.checkpoint.save("TestStage", {"__type__": "UnknownType", "data": {"key": "value"}})
        result = runner.run_stage("TestStage", lambda: None)
        assert isinstance(result, dict)
        assert result["key"] == "value"

    def test_no_envelope_returns_raw_dict(self):
        runner = CheckpointRunner(self.checkpoint, self.metrics)
        self.checkpoint.save("TestStage", {"legacy": "data"})
        result = runner.run_stage("TestStage", lambda: None)
        assert isinstance(result, dict)
        assert result["legacy"] == "data"

    def test_corrupt_data_raises_deserialization_error(self):
        runner = CheckpointRunner(
            self.checkpoint, self.metrics,
            type_registry={"VideoResult": VideoResult},
        )
        self.checkpoint.save("TestStage", {
            "__type__": "VideoResult",
            "data": {"bad_field": "no output_path"},
        })
        with pytest.raises(CheckpointDeserializationError):
            runner.run_stage("TestStage", lambda: None)

    def test_non_serializable_raises_serialization_error(self):
        runner = CheckpointRunner(self.checkpoint, self.metrics)
        with pytest.raises(CheckpointSerializationError):
            runner.run_stage("BadStage", lambda: 42)


# ============================================================
# 5  WAV Read Failures
# ============================================================


class TestWavReadFailures:

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            _read_wav("/nonexistent/audio.wav")

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.wav"
        p.write_bytes(b"")
        with pytest.raises(Exception):
            _read_wav(str(p))


# ============================================================
# 6  Debug Report Generation Under Failure
# ============================================================


class TestDebugReportGeneration:

    def test_generates_all_sections(self, tmp_path):
        config = ResolvedConfig.from_dict({"input_path": "/tmp/input.mp4"})
        metrics = PipelineMetrics("test-run")
        metric = metrics.begin_stage("FailingStage")
        metrics.end_stage(metric, "failed", "something broke")

        run_metadata = RunMetadata(
            run_id="test-run", pipeline_version="3.2.0",
            config_hash="abc123", timestamp="2024-01-01T00:00:00",
        )
        output_path = str(tmp_path / "debug_report.json")

        try:
            raise RuntimeError("test failure")
        except RuntimeError as e:
            DebugReportGenerator.generate(
                run_metadata=run_metadata, config=config,
                metrics=metrics, error=e,
                checkpoint_dir=str(tmp_path), output_path=output_path,
            )

        assert os.path.isfile(output_path)
        with open(output_path) as f:
            report = json.load(f)

        assert report["error"]["type"] == "RuntimeError"
        assert "test failure" in report["error"]["message"]
        assert report["pipeline"]["failed_stage"] == "FailingStage"
        assert report["pipeline"]["run_id"] == "test-run"
        assert "config" in report
        assert "environment" in report

    def test_report_with_no_metrics(self, tmp_path):
        config = ResolvedConfig.from_dict({})
        run_metadata = RunMetadata(
            run_id="x", pipeline_version="3.2.0",
            config_hash="", timestamp="",
        )
        output_path = str(tmp_path / "report.json")

        try:
            raise ValueError("boom")
        except ValueError as e:
            DebugReportGenerator.generate(
                run_metadata=run_metadata, config=config,
                metrics=None, error=e,
                checkpoint_dir=None, output_path=output_path,
            )

        with open(output_path) as f:
            report = json.load(f)
        assert report["pipeline"]["failed_stage"] is None
        assert report["pipeline"]["metrics"] is None


# ============================================================
# 7  FailureCaptureMiddleware Error Paths
# ============================================================


class TestFailureCaptureMiddlewareErrors:

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.run_folder = RunFolderManager(base_dir=str(tmp_path))
        self.run_folder.create("error_test")
        self.config = ResolvedConfig.from_dict({})
        self.run_metadata = RunMetadata(
            run_id="err", pipeline_version="3.2.0",
            config_hash="", timestamp="",
        )
        self.metrics = PipelineMetrics("err")

    def _make_middleware(self):
        return FailureCaptureMiddleware(
            run_metadata=self.run_metadata,
            config=self.config,
            metrics=self.metrics,
            run_folder=self.run_folder,
        )

    def test_reraises_original_exception_type(self):
        middleware = self._make_middleware()

        def fail():
            raise SourceSeparationError("separation failed")

        with pytest.raises(SourceSeparationError, match="separation failed"):
            middleware.run(fail)

    def test_debug_report_created_on_audio_error(self):
        middleware = self._make_middleware()

        def fail():
            raise AudioExtractionError("bad audio")

        with pytest.raises(AudioExtractionError):
            middleware.run(fail)

        assert os.path.isfile(self.run_folder.debug_report_path())


# ============================================================
# 8  Metrics Failure Tracking
# ============================================================


class TestMetricsFailureTracking:

    def test_failed_stage_recorded(self):
        metrics = PipelineMetrics("test-run")
        m = metrics.begin_stage("Stage1")
        metrics.end_stage(m, "failed", "something broke")

        summary = metrics.get_summary()
        assert summary["failed"] == 1
        assert summary["passed"] == 0
        assert summary["stages"][0]["status"] == "failed"
        assert summary["stages"][0]["error"] == "something broke"

    def test_mixed_success_and_failure(self):
        metrics = PipelineMetrics("test-run")
        m1 = metrics.begin_stage("Stage1")
        metrics.end_stage(m1, "success")
        m2 = metrics.begin_stage("Stage2")
        metrics.end_stage(m2, "failed", "error")

        summary = metrics.get_summary()
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["total_stages"] == 2
