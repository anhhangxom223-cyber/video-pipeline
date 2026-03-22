"""
test_integration.py — Integration tests.

Tests cross-module interactions: video-to-audio handoff via AudioInputAdapter,
CheckpointRunner + RunFolderManager round-trips, FailureCaptureMiddleware,
config → pipeline flow, and end-to-end DTO serialization.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile

import pytest
import numpy as np

from audio_pipeline import AudioInputContract, AudioResult
from video_pipeline import (
    MappingConfig,
    MappingEntry,
    VideoMetadata,
    VideoResult,
)
from config_loader import (
    ConfigLoader,
    PipelineMetrics,
    ResolvedConfig,
    RunMetadata,
    StageCheckpoint,
    stage_timer,
)
from core.runtime.run_folder_manager import RunFolderManager
from core.runtime.checkpoint_runner import CheckpointRunner
from core.observability.failure_middleware import FailureCaptureMiddleware
from core.observability.metrics_instrumentation import instrument_stage


# ============================================================
# Helpers
# ============================================================


def _adapt_video_to_audio(video_result: VideoResult) -> AudioInputContract:
    """Inline AudioInputAdapter per freeze spec: VideoResult → AudioInputContract."""
    if not isinstance(video_result, VideoResult):
        raise TypeError("video_result must be VideoResult")
    return AudioInputContract(
        silent_video_path=video_result.output_path,
        translated_srt_path=video_result.subtitle_path or "",
        metadata=video_result.metadata.to_dict(),
    )


def _make_video_result(
    path="/tmp/silent_video.mp4",
    srt="/tmp/translated.srt",
):
    meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)
    return VideoResult(output_path=path, subtitle_path=srt, metadata=meta)


# ============================================================
# 1  Video-to-Audio Handoff
# ============================================================


class TestVideoToAudioHandoff:
    """
    Freeze spec: VideoResult is the contract boundary between video and audio.
    AudioInputAdapter converts VideoResult → AudioInputContract.
    """

    def test_produces_audio_input_contract(self):
        vr = _make_video_result()
        aic = _adapt_video_to_audio(vr)
        assert isinstance(aic, AudioInputContract)
        assert aic.silent_video_path == "/tmp/silent_video.mp4"
        assert aic.translated_srt_path == "/tmp/translated.srt"

    def test_passes_metadata(self):
        vr = _make_video_result()
        aic = _adapt_video_to_audio(vr)
        assert aic.metadata["width"] == 320
        assert aic.metadata["height"] == 240
        assert aic.metadata["fps"] == 30.0

    def test_rejects_non_video_result(self):
        with pytest.raises(TypeError):
            _adapt_video_to_audio({"not": "a VideoResult"})

    def test_none_subtitle_path_becomes_empty(self):
        meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)
        vr = VideoResult(output_path="/tmp/video.mp4", subtitle_path=None, metadata=meta)
        aic = _adapt_video_to_audio(vr)
        assert aic.translated_srt_path == ""

    def test_both_dtos_remain_frozen(self):
        vr = _make_video_result()
        aic = _adapt_video_to_audio(vr)
        with pytest.raises(Exception):
            vr.output_path = "changed"
        with pytest.raises(Exception):
            aic.silent_video_path = "changed"


# ============================================================
# 2  Config Resolution
# ============================================================


class TestConfigResolution:

    def test_defaults_resolve(self):
        resolved = ConfigLoader.resolve()
        assert "engine_mode" in resolved
        assert resolved["engine_mode"] == "HSV"

    def test_cli_override(self):
        resolved = ConfigLoader.resolve(cli_args={"engine_mode": "LAB"})
        assert resolved["engine_mode"] == "LAB"

    def test_json_plus_cli(self, tmp_path):
        p = tmp_path / "config.json"
        p.write_text(json.dumps({"engine_mode": "LAB", "target_language": "es"}))
        config = ConfigLoader.resolve_from_paths(
            json_path=str(p),
            cli_overrides={"engine_mode": "HSV"},
        )
        assert config.values["engine_mode"] == "HSV"
        assert config.values["target_language"] == "es"


# ============================================================
# 3  CheckpointRunner + RunFolderManager
# ============================================================


class TestCheckpointIntegration:
    """
    CheckpointRunner persists stage results and resumes on re-run.
    RunFolderManager provides the directory structure.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.run_folder = RunFolderManager(base_dir=str(tmp_path))
        self.run_folder.create("integration_test")
        self.metrics = PipelineMetrics("integration_test")
        self.checkpoint = StageCheckpoint(self.run_folder.checkpoint_path())

    def test_video_result_round_trip(self):
        runner = CheckpointRunner(
            self.checkpoint, self.metrics,
            type_registry={"VideoResult": VideoResult},
        )
        expected = _make_video_result()
        result1 = runner.run_stage("Video", lambda: expected)
        assert result1 == expected

        result2 = runner.run_stage("Video", lambda: None)
        assert isinstance(result2, VideoResult)
        assert result2.output_path == expected.output_path

    def test_audio_result_round_trip(self):
        runner = CheckpointRunner(
            self.checkpoint, self.metrics,
            type_registry={"AudioResult": AudioResult},
        )
        expected = AudioResult(audio_path="/tmp/audio.wav", metadata={"duration": 10.0})
        result1 = runner.run_stage("Audio", lambda: expected)
        assert result1 == expected

        result2 = runner.run_stage("Audio", lambda: None)
        assert isinstance(result2, AudioResult)
        assert result2.audio_path == expected.audio_path

    def test_metrics_records_stages(self):
        runner = CheckpointRunner(self.checkpoint, self.metrics)
        meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)
        runner.run_stage(
            "Stage1",
            lambda: VideoResult(output_path="/tmp/a.mp4", subtitle_path=None, metadata=meta),
        )
        summary = self.metrics.get_summary()
        assert summary["total_stages"] == 1
        assert summary["passed"] == 1
        assert summary["failed"] == 0


# ============================================================
# 4  FailureCaptureMiddleware
# ============================================================


class TestFailureCaptureMiddleware:

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.run_folder = RunFolderManager(base_dir=str(tmp_path))
        self.run_folder.create("middleware_test")
        self.metrics = PipelineMetrics("middleware_test")
        self.config = ResolvedConfig.from_dict({"key": "value"})
        self.run_metadata = RunMetadata(
            run_id="test", pipeline_version="3.2.0",
            config_hash="abc", timestamp="2024-01-01T00:00:00",
        )

    def _make_middleware(self):
        return FailureCaptureMiddleware(
            run_metadata=self.run_metadata,
            config=self.config,
            metrics=self.metrics,
            run_folder=self.run_folder,
        )

    def test_success_passes_through(self):
        result = self._make_middleware().run(lambda: {"status": "ok"})
        assert result["status"] == "ok"

    def test_failure_generates_debug_report(self):
        middleware = self._make_middleware()
        with pytest.raises(RuntimeError):
            middleware.run(lambda: (_ for _ in ()).throw(RuntimeError("Pipeline exploded")))

        report_path = self.run_folder.debug_report_path()
        assert os.path.isfile(report_path)
        with open(report_path) as f:
            report = json.load(f)
        assert report["error"]["type"] == "RuntimeError"
        assert "Pipeline exploded" in report["error"]["message"]
        assert "pipeline" in report
        assert "config" in report
        assert "environment" in report


# ============================================================
# 5  Mock Frame Generation
# ============================================================


class TestMockFrameGeneration:

    @staticmethod
    def _generate_frames(width=320, height=240, count=5):
        for i in range(count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 25) % 256
            frame[:, :, 1] = (i * 50) % 256
            frame[:, :, 2] = (i * 75) % 256
            yield frame

    def test_frame_shape_and_dtype(self):
        frames = list(self._generate_frames(width=320, height=240, count=5))
        assert len(frames) == 5
        for frame in frames:
            assert frame.shape == (240, 320, 3)
            assert frame.dtype == np.uint8

    def test_frames_differ(self):
        frames = list(self._generate_frames(width=64, height=64, count=3))
        assert not np.array_equal(frames[0], frames[1])


# ============================================================
# 6  End-to-End Serialization
# ============================================================


class TestEndToEndSerialization:

    def test_video_result_roundtrip(self):
        meta = VideoMetadata(width=1920, height=1080, fps=60.0, frame_count=3600)
        original = VideoResult(
            output_path="/output/video.mp4",
            subtitle_path="/output/subs.srt",
            metadata=meta,
        )
        assert VideoResult.from_dict(original.to_dict()) == original

    def test_audio_result_roundtrip(self):
        original = AudioResult(
            audio_path="/output/audio.wav",
            metadata={"sample_rate": 44100, "channels": 1},
        )
        assert AudioResult.from_dict(original.to_dict()) == original

    def test_audio_input_contract_roundtrip(self):
        original = AudioInputContract(
            silent_video_path="/tmp/video.mp4",
            translated_srt_path="/tmp/subs.srt",
            metadata={"fps": 30},
        )
        assert AudioInputContract.from_dict(original.to_dict()) == original

    def test_mapping_config_roundtrip(self):
        original = MappingConfig(
            version="1.0.0", engine_mode="HSV",
            mappings=(MappingEntry(source_range=(0, 50), target_hue_range=(100, 150)),),
        )
        assert MappingConfig.from_dict(original.to_dict()) == original
