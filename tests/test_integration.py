"""
test_integration.py — Integration tests.

Tests cross-module interactions: video-to-audio handoff via AudioInputAdapter,
full pipeline orchestration with CheckpointRunner and FailureCaptureMiddleware,
config → pipeline → result end-to-end flow, and observability integration.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    generate_run_id,
    stage_timer,
)
from core.runtime.run_folder_manager import RunFolderManager
from core.runtime.checkpoint_runner import CheckpointRunner
from core.observability.failure_middleware import FailureCaptureMiddleware
from core.observability.metrics_instrumentation import instrument_stage


# ============================================================
# Section 1: Video-to-Audio Handoff
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


class TestVideoToAudioHandoff(unittest.TestCase):
    """
    Freeze spec: VideoResult is the contract boundary between video and audio.
    AudioInputAdapter converts VideoResult → AudioInputContract.
    """

    def _make_video_result(self):
        meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)
        return VideoResult(
            output_path="/tmp/silent_video.mp4",
            subtitle_path="/tmp/translated.srt",
            metadata=meta,
        )

    def test_adapter_produces_audio_input_contract(self):
        vr = self._make_video_result()
        aic = _adapt_video_to_audio(vr)

        self.assertIsInstance(aic, AudioInputContract)
        self.assertEqual(aic.silent_video_path, "/tmp/silent_video.mp4")
        self.assertEqual(aic.translated_srt_path, "/tmp/translated.srt")

    def test_adapter_passes_metadata(self):
        vr = self._make_video_result()
        aic = _adapt_video_to_audio(vr)

        self.assertIn("width", aic.metadata)
        self.assertEqual(aic.metadata["width"], 320)

    def test_adapter_rejects_non_video_result(self):
        with self.assertRaises(TypeError):
            _adapt_video_to_audio({"not": "a VideoResult"})

    def test_adapter_handles_none_subtitle_path(self):
        meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)
        vr = VideoResult(output_path="/tmp/video.mp4", subtitle_path=None, metadata=meta)
        aic = _adapt_video_to_audio(vr)
        self.assertEqual(aic.translated_srt_path, "")

    def test_handoff_dto_immutability(self):
        """Both VideoResult and AudioInputContract must remain frozen."""
        vr = self._make_video_result()
        aic = _adapt_video_to_audio(vr)

        with self.assertRaises(Exception):
            vr.output_path = "changed"

        with self.assertRaises(Exception):
            aic.silent_video_path = "changed"


# ============================================================
# Section 2: Config → Pipeline Factory
# ============================================================


class TestConfigToPipelineFactory(unittest.TestCase):
    """ConfigLoader resolves config and produces valid ResolvedConfig."""

    def test_default_config_resolves(self):
        resolved = ConfigLoader.resolve()
        self.assertIn("engine_mode", resolved)
        self.assertEqual(resolved["engine_mode"], "HSV")

    def test_config_resolution_with_cli(self):
        resolved = ConfigLoader.resolve(cli_args={"fps": "24.0"})
        self.assertEqual(resolved["fps"], "24.0")

    def test_resolve_from_paths_with_json(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"fps": "60.0", "engine_mode": "LAB"}, f)
            path = f.name

        try:
            config = ConfigLoader.resolve_from_paths(
                json_path=path,
                cli_overrides={"engine_mode": "HSV"},
            )
            # CLI overrides JSON
            self.assertEqual(config.values["engine_mode"], "HSV")
            # JSON overrides defaults
            self.assertEqual(config.values["fps"], "60.0")
        finally:
            os.unlink(path)


# ============================================================
# Section 3: CheckpointRunner + RunFolderManager Integration
# ============================================================


class TestCheckpointIntegration(unittest.TestCase):
    """
    CheckpointRunner persists stage results and resumes on re-run.
    RunFolderManager provides the directory structure.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.run_folder = RunFolderManager(base_dir=self.tmpdir)
        self.run_folder.create("integration_test")
        self.metrics = PipelineMetrics("integration_test")
        self.checkpoint = StageCheckpoint(self.run_folder.checkpoint_path())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_checkpoint_round_trip_video_result(self):
        type_registry = {"VideoResult": VideoResult}
        runner = CheckpointRunner(self.checkpoint, self.metrics, type_registry=type_registry)

        meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)
        expected = VideoResult(
            output_path="/tmp/out.mp4",
            subtitle_path="/tmp/subs.srt",
            metadata=meta,
        )

        def stage_fn():
            return expected

        # First run: executes and saves checkpoint
        result1 = runner.run_stage("Video", stage_fn)
        self.assertEqual(result1, expected)

        # Second run: resumes from checkpoint
        result2 = runner.run_stage("Video", lambda: None)
        self.assertIsInstance(result2, VideoResult)
        self.assertEqual(result2.output_path, expected.output_path)

    def test_checkpoint_round_trip_audio_result(self):
        type_registry = {"AudioResult": AudioResult}
        runner = CheckpointRunner(self.checkpoint, self.metrics, type_registry=type_registry)

        expected = AudioResult(audio_path="/tmp/audio.wav", metadata={"duration": 10.0})

        result1 = runner.run_stage("Audio", lambda: expected)
        self.assertEqual(result1, expected)

        result2 = runner.run_stage("Audio", lambda: None)
        self.assertIsInstance(result2, AudioResult)
        self.assertEqual(result2.audio_path, expected.audio_path)

    def test_metrics_records_stages(self):
        runner = CheckpointRunner(self.checkpoint, self.metrics)
        meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)

        runner.run_stage(
            "Stage1",
            lambda: VideoResult(output_path="/tmp/a.mp4", subtitle_path=None, metadata=meta),
        )

        summary = self.metrics.get_summary()
        self.assertEqual(summary["total_stages"], 1)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 0)


# ============================================================
# Section 4: FailureCaptureMiddleware Integration
# ============================================================


class TestFailureCaptureMiddleware(unittest.TestCase):
    """
    FailureCaptureMiddleware wraps pipeline execution and generates
    a debug report on failure.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.run_folder = RunFolderManager(base_dir=self.tmpdir)
        self.run_folder.create("failure_test")
        self.metrics = PipelineMetrics("failure_test")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_success_passes_through(self):
        config = ResolvedConfig.from_dict({"key": "value"})
        run_metadata = RunMetadata(
            run_id="test", pipeline_version="3.2.0",
            config_hash="abc", timestamp="2024-01-01T00:00:00",
        )
        middleware = FailureCaptureMiddleware(
            run_metadata=run_metadata,
            config=config,
            metrics=self.metrics,
            run_folder=self.run_folder,
        )

        result = middleware.run(lambda: {"status": "ok"})
        self.assertEqual(result["status"], "ok")

    def test_failure_generates_debug_report(self):
        config = ResolvedConfig.from_dict({"key": "value"})
        run_metadata = RunMetadata(
            run_id="test", pipeline_version="3.2.0",
            config_hash="abc", timestamp="2024-01-01T00:00:00",
        )
        middleware = FailureCaptureMiddleware(
            run_metadata=run_metadata,
            config=config,
            metrics=self.metrics,
            run_folder=self.run_folder,
        )

        def failing_pipeline():
            raise RuntimeError("Pipeline exploded")

        with self.assertRaises(RuntimeError):
            middleware.run(failing_pipeline)

        report_path = self.run_folder.debug_report_path()
        self.assertTrue(os.path.isfile(report_path))

        with open(report_path) as f:
            report = json.load(f)

        self.assertEqual(report["error"]["type"], "RuntimeError")
        self.assertIn("Pipeline exploded", report["error"]["message"])
        self.assertIn("pipeline", report)
        self.assertIn("config", report)
        self.assertIn("environment", report)


# ============================================================
# Section 5: Mock Frame Generation
# ============================================================


class TestMockFrameGeneration(unittest.TestCase):
    """Verify numpy frame arrays match freeze spec constraints."""

    @staticmethod
    def _generate_frames(width=320, height=240, count=5):
        for i in range(count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 25) % 256
            frame[:, :, 1] = (i * 50) % 256
            frame[:, :, 2] = (i * 75) % 256
            yield frame

    def test_frame_shape(self):
        frames = list(self._generate_frames(width=320, height=240, count=5))
        self.assertEqual(len(frames), 5)
        for frame in frames:
            self.assertEqual(frame.shape, (240, 320, 3))
            self.assertEqual(frame.dtype, np.uint8)

    def test_frames_are_different(self):
        frames = list(self._generate_frames(width=64, height=64, count=3))
        self.assertFalse(np.array_equal(frames[0], frames[1]))


# ============================================================
# Section 6: End-to-End Serialization
# ============================================================


class TestEndToEndSerialization(unittest.TestCase):
    """Video and audio results serialize through checkpoint envelope."""

    def test_video_result_to_dict_from_dict(self):
        meta = VideoMetadata(width=1920, height=1080, fps=60.0, frame_count=3600)
        original = VideoResult(
            output_path="/output/video.mp4",
            subtitle_path="/output/subs.srt",
            metadata=meta,
        )
        data = original.to_dict()
        restored = VideoResult.from_dict(data)
        self.assertEqual(original, restored)

    def test_audio_result_to_dict_from_dict(self):
        original = AudioResult(
            audio_path="/output/audio.wav",
            metadata={"sample_rate": 44100, "channels": 1},
        )
        data = original.to_dict()
        restored = AudioResult.from_dict(data)
        self.assertEqual(original, restored)

    def test_audio_input_contract_to_dict_from_dict(self):
        original = AudioInputContract(
            silent_video_path="/tmp/video.mp4",
            translated_srt_path="/tmp/subs.srt",
            metadata={"fps": 30},
        )
        data = original.to_dict()
        restored = AudioInputContract.from_dict(data)
        self.assertEqual(original, restored)


if __name__ == "__main__":
    unittest.main()
