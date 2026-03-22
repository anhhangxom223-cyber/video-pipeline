"""
test_functional.py — Functional unit tests.

Tests individual module behavior: DTO construction, validation, serialization,
ConfigLoader resolution, schema validation, WAV I/O, checkpoint runner,
run folder management, and metrics instrumentation.
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
    _read_wav,
    _write_wav,
)
from video_pipeline import (
    MappingConfig,
    MappingEntry,
    ROIBox,
    VideoMetadata,
    VideoOutputContract,
    VideoResult,
)
from config_loader import (
    ConfigLoader,
    PipelineContext,
    PipelineContextFactory,
    PipelineMetrics,
    ResolvedConfig,
    RunMetadata,
    StageCheckpoint,
    generate_run_id,
    compute_config_hash,
    compute_file_hash,
    create_run_metadata,
    save_run_metadata,
    stage_timer,
)
from core.runtime.run_folder_manager import RunFolderManager
from core.runtime.checkpoint_runner import (
    CheckpointRunner,
    CheckpointSerializationError,
    CheckpointDeserializationError,
)
from core.observability.metrics_instrumentation import instrument_stage


# ============================================================
# Section 1: Audio DTO Construction & Validation
# ============================================================


class TestAudioDTOConstruction(unittest.TestCase):
    """Validate correct construction of all 15 audio DTOs."""

    def test_time_range(self):
        tr = TimeRange(start=0.0, end=1.5)
        self.assertEqual(tr.start, 0.0)
        self.assertEqual(tr.end, 1.5)

    def test_pitch_frame(self):
        pf = PitchFrame(time_offset=0.1, frequency_hz=440.0, confidence=0.95)
        self.assertAlmostEqual(pf.frequency_hz, 440.0)
        self.assertAlmostEqual(pf.confidence, 0.95)

    def test_rms_frame(self):
        rf = RMSFrame(time_offset=0.0, rms_value=0.3)
        self.assertAlmostEqual(rf.rms_value, 0.3)

    def test_segment_prosody(self):
        sp = SegmentProsody(
            speech_rate=1.2,
            rms_frames=(RMSFrame(time_offset=0.0, rms_value=0.1),),
            pitch_frames=(PitchFrame(time_offset=0.0, frequency_hz=220.0, confidence=0.8),),
        )
        self.assertEqual(len(sp.rms_frames), 1)
        self.assertEqual(len(sp.pitch_frames), 1)

    def test_asr_segment(self):
        seg = ASRSegment(
            segment_id=0,
            time_range=TimeRange(start=0.0, end=2.0),
            transcript="hello world",
            avg_logprob=-0.3,
            no_speech_prob=0.05,
            prosody=SegmentProsody(
                speech_rate=1.0,
                rms_frames=(),
                pitch_frames=(),
            ),
        )
        self.assertEqual(seg.segment_id, 0)
        self.assertEqual(seg.transcript, "hello world")

    def test_extracted_audio(self):
        ea = ExtractedAudio(
            audio_path="/tmp/audio.wav",
            sample_rate=44100,
            channels=1,
            duration_seconds=3.0,
        )
        self.assertEqual(ea.sample_rate, 44100)

    def test_separated_audio(self):
        sa = SeparatedAudio(
            voice_audio_path="/tmp/voice.wav",
            background_audio_path="/tmp/bg.wav",
        )
        self.assertEqual(sa.voice_audio_path, "/tmp/voice.wav")

    def test_voice_analysis_result(self):
        var = VoiceAnalysisResult(asr_segments=())
        self.assertEqual(len(var.asr_segments), 0)

    def test_aligned_segment(self):
        a = AlignedSegment(
            segment_id=0,
            time_range=TimeRange(start=0.0, end=1.0),
            original_text="hello",
            translated_text="hola",
        )
        self.assertEqual(a.translated_text, "hola")

    def test_tts_output_segment(self):
        t = TTSOutputSegment(
            segment_id=0,
            time_range=TimeRange(start=0.0, end=1.0),
            audio_path="/tmp/tts.wav",
            duration_seconds=1.0,
            pitch_shift_delta=0.0,
        )
        self.assertAlmostEqual(t.duration_seconds, 1.0)

    def test_duration_aligned_segment(self):
        d = DurationAlignedSegment(
            segment_id=0,
            time_range=TimeRange(start=0.0, end=1.0),
            audio_path="/tmp/dur.wav",
            adjusted_duration=0.95,
            stretch_ratio=1.05,
        )
        self.assertAlmostEqual(d.stretch_ratio, 1.05)

    def test_rms_processed_segment(self):
        r = RMSProcessedSegment(
            segment_id=0,
            time_range=TimeRange(start=0.0, end=1.0),
            audio_path="/tmp/rms.wav",
            rms_adjustment_delta=0.02,
        )
        self.assertAlmostEqual(r.rms_adjustment_delta, 0.02)

    def test_mixed_audio(self):
        m = MixedAudio(
            audio_path="/tmp/mixed.wav",
            integrated_lufs=-14.0,
            peak_db=-1.0,
        )
        self.assertAlmostEqual(m.integrated_lufs, -14.0)

    def test_audio_input_contract(self):
        c = AudioInputContract(
            silent_video_path="/tmp/video.mp4",
            translated_srt_path="/tmp/subs.srt",
            metadata={"fps": 30},
        )
        self.assertEqual(c.silent_video_path, "/tmp/video.mp4")

    def test_audio_result(self):
        r = AudioResult(audio_path="/tmp/result.wav", metadata={"duration": 10.0})
        self.assertEqual(r.audio_path, "/tmp/result.wav")


# ============================================================
# Section 2: Video DTO Construction & Validation
# ============================================================


class TestVideoDTOConstruction(unittest.TestCase):
    """Validate correct construction and post-init validation of video DTOs."""

    def test_roi_box_valid(self):
        roi = ROIBox(x=10, y=20, width=100, height=50)
        self.assertEqual(roi.to_dict(), {"x": 10, "y": 20, "width": 100, "height": 50})

    def test_roi_box_negative_coordinates_rejected(self):
        with self.assertRaises(ValueError):
            ROIBox(x=-1, y=0, width=100, height=50)

    def test_roi_box_zero_dimension_rejected(self):
        with self.assertRaises(ValueError):
            ROIBox(x=0, y=0, width=0, height=50)

    def test_roi_box_roundtrip(self):
        original = ROIBox(x=5, y=10, width=200, height=100)
        restored = ROIBox.from_dict(original.to_dict())
        self.assertEqual(original, restored)

    def test_mapping_entry_valid(self):
        entry = MappingEntry(source_range=(0, 50), target_hue_range=(100, 150))
        self.assertEqual(entry.source_range, (0, 50))

    def test_mapping_entry_invalid_range(self):
        with self.assertRaises(ValueError):
            MappingEntry(source_range=(100, 50), target_hue_range=(0, 50))

    def test_mapping_config_valid(self):
        cfg = MappingConfig(
            version="1.0.0",
            engine_mode="HSV",
            mappings=(MappingEntry(source_range=(0, 50), target_hue_range=(100, 150)),),
        )
        self.assertEqual(cfg.version, "1.0.0")

    def test_mapping_config_invalid_version(self):
        with self.assertRaises(ValueError):
            MappingConfig(
                version="not-semver",
                engine_mode="HSV",
                mappings=(MappingEntry(source_range=(0, 50), target_hue_range=(100, 150)),),
            )

    def test_mapping_config_invalid_engine_mode(self):
        with self.assertRaises(ValueError):
            MappingConfig(
                version="1.0.0",
                engine_mode="RGB",
                mappings=(MappingEntry(source_range=(0, 50), target_hue_range=(100, 150)),),
            )

    def test_mapping_config_roundtrip(self):
        original = MappingConfig(
            version="2.0.0",
            engine_mode="LAB",
            mappings=(MappingEntry(source_range=(10, 20), target_hue_range=(30, 40)),),
        )
        restored = MappingConfig.from_dict(original.to_dict())
        self.assertEqual(original, restored)

    def test_video_metadata_valid(self):
        meta = VideoMetadata(width=1920, height=1080, fps=30.0, frame_count=900)
        self.assertEqual(meta.width, 1920)

    def test_video_metadata_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            VideoMetadata(width=0, height=1080, fps=30.0, frame_count=100)

    def test_video_result_valid(self):
        meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)
        result = VideoResult(output_path="/tmp/out.mp4", subtitle_path=None, metadata=meta)
        self.assertEqual(result.output_path, "/tmp/out.mp4")

    def test_video_result_empty_path_rejected(self):
        meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)
        with self.assertRaises(ValueError):
            VideoResult(output_path="", subtitle_path=None, metadata=meta)

    def test_video_result_roundtrip(self):
        meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)
        original = VideoResult(output_path="/tmp/test.mp4", subtitle_path="/tmp/subs.srt", metadata=meta)
        restored = VideoResult.from_dict(original.to_dict())
        self.assertEqual(original, restored)

    def test_video_output_contract(self):
        meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)
        result = VideoResult(output_path="/tmp/out.mp4", subtitle_path=None, metadata=meta)
        contract = VideoOutputContract(result=result)
        self.assertIsInstance(contract.result, VideoResult)


# ============================================================
# Section 3: ConfigLoader
# ============================================================


class TestConfigLoader(unittest.TestCase):
    """ConfigLoader resolution: defaults ← JSON ← CLI."""

    def test_resolve_defaults_only(self):
        resolved = ConfigLoader.resolve()
        self.assertIn("engine_mode", resolved)
        self.assertEqual(resolved["engine_mode"], "HSV")
        self.assertEqual(resolved["face_fusion_enabled"], "false")

    def test_resolve_cli_overrides_defaults(self):
        resolved = ConfigLoader.resolve(cli_args={"engine_mode": "LAB"})
        self.assertEqual(resolved["engine_mode"], "LAB")

    def test_resolve_json_overridden_by_cli(self):
        resolved = ConfigLoader.resolve(
            json_config={"engine_mode": "LAB"},
            cli_args={"engine_mode": "HSV"},
        )
        self.assertEqual(resolved["engine_mode"], "HSV")

    def test_resolve_returns_sorted_dict(self):
        resolved = ConfigLoader.resolve()
        keys = list(resolved.keys())
        self.assertEqual(keys, sorted(keys))

    def test_resolved_config_immutable(self):
        cfg = ResolvedConfig(configuration_parameters={"key": "value"})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg.configuration_parameters = {}

    def test_resolved_config_from_dict(self):
        cfg = ResolvedConfig.from_dict({"a": "1", "b": "2"})
        self.assertEqual(cfg.values["a"], "1")

    def test_load_json_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"engine_mode": "LAB"}, f)
            path = f.name

        try:
            loaded = ConfigLoader.load_json(path)
            self.assertEqual(loaded["engine_mode"], "LAB")
        finally:
            os.unlink(path)

    def test_load_json_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            ConfigLoader.load_json("/nonexistent/config.json")


# ============================================================
# Section 4: PipelineContext
# ============================================================


class TestPipelineContext(unittest.TestCase):

    def test_factory_creates_context(self):
        cfg = ResolvedConfig.from_dict({"key": "value"})
        ctx = PipelineContextFactory.create(cfg)
        self.assertIsInstance(ctx, PipelineContext)
        self.assertEqual(ctx.config.values["key"], "value")

    def test_factory_rejects_non_resolved_config(self):
        with self.assertRaises(TypeError):
            PipelineContextFactory.create({"key": "value"})


# ============================================================
# Section 5: WAV I/O
# ============================================================


class TestWavIO(unittest.TestCase):
    """WAV read/write roundtrip using audio_pipeline utilities."""

    def test_write_and_read_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            samples = np.sin(np.linspace(0, 2 * np.pi, 44100)).astype(np.float64)
            _write_wav(path, samples, sample_rate=44100)

            read_samples, rate = _read_wav(path)
            self.assertEqual(rate, 44100)
            self.assertEqual(len(read_samples), 44100)
            # Quantization error from int16 roundtrip
            np.testing.assert_allclose(read_samples, samples, atol=1e-4)
        finally:
            os.unlink(path)


# ============================================================
# Section 6: Versioning & Run Metadata
# ============================================================


class TestVersioning(unittest.TestCase):

    def test_generate_run_id_deterministic(self):
        config = {"a": "1", "b": "2"}
        id1 = generate_run_id(config)
        id2 = generate_run_id(config)
        self.assertEqual(id1, id2)
        self.assertTrue(id1.startswith("run_"))

    def test_different_configs_produce_different_ids(self):
        id1 = generate_run_id({"a": "1"})
        id2 = generate_run_id({"a": "2"})
        self.assertNotEqual(id1, id2)

    def test_compute_config_hash(self):
        h = compute_config_hash({"key": "value"})
        self.assertEqual(len(h), 16)

    def test_compute_file_hash_missing_file(self):
        self.assertEqual(compute_file_hash("/nonexistent/file"), "")

    def test_create_run_metadata(self):
        meta = create_run_metadata({"a": "1"})
        self.assertTrue(meta.run_id.startswith("run_"))
        self.assertEqual(meta.pipeline_version, "3.2.0")

    def test_save_run_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = create_run_metadata({"a": "1"})
            path = save_run_metadata(meta, tmpdir)
            self.assertTrue(os.path.isfile(path))

            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["run_id"], meta.run_id)


# ============================================================
# Section 7: StageCheckpoint
# ============================================================


class TestStageCheckpoint(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.checkpoint = StageCheckpoint(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_save_and_load(self):
        self.checkpoint.save("stage1", {"result": "ok"})
        data = self.checkpoint.load("stage1")
        self.assertEqual(data["result"], "ok")

    def test_exists(self):
        self.assertFalse(self.checkpoint.exists("stage1"))
        self.checkpoint.save("stage1", {"x": 1})
        self.assertTrue(self.checkpoint.exists("stage1"))

    def test_load_missing_returns_none(self):
        self.assertIsNone(self.checkpoint.load("nonexistent"))

    def test_clear(self):
        self.checkpoint.save("s1", {"a": 1})
        self.checkpoint.save("s2", {"b": 2})
        self.checkpoint.clear()
        self.assertFalse(self.checkpoint.exists("s1"))
        self.assertFalse(self.checkpoint.exists("s2"))


# ============================================================
# Section 8: CheckpointRunner
# ============================================================


class TestCheckpointRunner(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.checkpoint = StageCheckpoint(self.tmpdir)
        self.metrics = PipelineMetrics("test-run")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_run_stage_fresh_execution(self):
        runner = CheckpointRunner(self.checkpoint, self.metrics)

        def stage_fn():
            meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)
            return VideoResult(output_path="/tmp/out.mp4", subtitle_path=None, metadata=meta)

        result = runner.run_stage("TestStage", stage_fn)
        self.assertIsInstance(result, VideoResult)
        self.assertTrue(self.checkpoint.exists("TestStage"))

    def test_run_stage_resumes_from_checkpoint(self):
        type_registry = {"VideoResult": VideoResult}
        runner = CheckpointRunner(self.checkpoint, self.metrics, type_registry=type_registry)

        meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)
        original = VideoResult(output_path="/tmp/out.mp4", subtitle_path=None, metadata=meta)

        # Manually save checkpoint
        self.checkpoint.save("TestStage", {
            "__type__": "VideoResult",
            "data": original.to_dict(),
        })

        call_count = 0

        def stage_fn():
            nonlocal call_count
            call_count += 1
            return None

        result = runner.run_stage("TestStage", stage_fn)
        self.assertEqual(call_count, 0)  # Should not have called stage_fn
        self.assertIsInstance(result, VideoResult)
        self.assertEqual(result.output_path, "/tmp/out.mp4")

    def test_serialization_error_on_missing_to_dict(self):
        runner = CheckpointRunner(self.checkpoint, self.metrics)

        def stage_fn():
            return "plain string"

        with self.assertRaises(CheckpointSerializationError):
            runner.run_stage("BadStage", stage_fn)


# ============================================================
# Section 9: RunFolderManager
# ============================================================


class TestRunFolderManager(unittest.TestCase):

    def test_create_and_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RunFolderManager(base_dir=tmpdir)
            run_path = mgr.create("test_run")

            self.assertTrue(os.path.isdir(run_path))
            self.assertTrue(os.path.isdir(mgr.artifacts_dir))
            self.assertTrue(os.path.isdir(mgr.checkpoints_dir))
            self.assertTrue(os.path.isdir(mgr.logs_dir))

            self.assertIn("test_run", mgr.artifact_path("test.mp4"))
            self.assertIn("checkpoints", mgr.checkpoint_path())
            self.assertIn("pipeline.log", mgr.log_path())
            self.assertIn("debug_report.json", mgr.debug_report_path())


# ============================================================
# Section 10: Metrics Instrumentation
# ============================================================


class TestMetricsInstrumentation(unittest.TestCase):

    def test_instrument_stage_records_success(self):
        metrics = PipelineMetrics("test-run")

        @instrument_stage(metrics, "MyStage")
        def my_fn():
            return 42

        result = my_fn()
        self.assertEqual(result, 42)

        summary = metrics.get_summary()
        self.assertEqual(summary["total_stages"], 1)
        self.assertEqual(summary["passed"], 1)

    def test_instrument_stage_records_failure(self):
        metrics = PipelineMetrics("test-run")

        @instrument_stage(metrics, "FailStage")
        def bad_fn():
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            bad_fn()

        summary = metrics.get_summary()
        self.assertEqual(summary["failed"], 1)


# ============================================================
# Section 11: stage_timer Context Manager
# ============================================================


class TestStageTimer(unittest.TestCase):

    def test_stage_timer_records_elapsed(self):
        with stage_timer("TestStage") as metrics:
            _ = sum(range(100))

        self.assertIn("elapsed_ms", metrics)
        self.assertGreaterEqual(metrics["elapsed_ms"], 0.0)
        self.assertEqual(metrics["stage"], "TestStage")


if __name__ == "__main__":
    unittest.main()
