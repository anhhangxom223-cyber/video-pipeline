"""
test_functional.py — Functional unit tests.

Tests individual module behaviour: DTO construction & validation, serialization
round-trips, ConfigLoader resolution, schema validation, WAV I/O, checkpoint
runner, run folder management, and metrics instrumentation.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import tempfile

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
)
from core.observability.metrics_instrumentation import instrument_stage


# ============================================================
# 1  Audio DTO Construction (ADM-01 → ADM-15)
# ============================================================


class TestAudioDTOConstruction:
    """Validate correct construction of all 15 audio DTOs."""

    def test_time_range(self):
        tr = TimeRange(start=0.0, end=1.5)
        assert tr.start == 0.0
        assert tr.end == 1.5

    def test_pitch_frame(self):
        pf = PitchFrame(time_offset=0.1, frequency_hz=440.0, confidence=0.95)
        assert pf.frequency_hz == pytest.approx(440.0)
        assert pf.confidence == pytest.approx(0.95)

    def test_rms_frame(self):
        rf = RMSFrame(time_offset=0.0, rms_value=0.3)
        assert rf.rms_value == pytest.approx(0.3)

    def test_segment_prosody(self):
        sp = SegmentProsody(
            speech_rate=1.2,
            rms_frames=(RMSFrame(time_offset=0.0, rms_value=0.1),),
            pitch_frames=(PitchFrame(time_offset=0.0, frequency_hz=220.0, confidence=0.8),),
        )
        assert len(sp.rms_frames) == 1
        assert len(sp.pitch_frames) == 1

    def test_asr_segment(self):
        seg = ASRSegment(
            segment_id=0,
            time_range=TimeRange(start=0.0, end=2.0),
            transcript="hello world",
            avg_logprob=-0.3,
            no_speech_prob=0.05,
            prosody=SegmentProsody(speech_rate=1.0, rms_frames=(), pitch_frames=()),
        )
        assert seg.segment_id == 0
        assert seg.transcript == "hello world"

    def test_extracted_audio(self):
        ea = ExtractedAudio(audio_path="/tmp/audio.wav", sample_rate=44100, channels=1, duration_seconds=3.0)
        assert ea.sample_rate == 44100

    def test_separated_audio(self):
        sa = SeparatedAudio(voice_audio_path="/tmp/voice.wav", background_audio_path="/tmp/bg.wav")
        assert sa.voice_audio_path == "/tmp/voice.wav"

    def test_voice_analysis_result(self):
        var = VoiceAnalysisResult(asr_segments=())
        assert len(var.asr_segments) == 0

    def test_aligned_segment(self):
        a = AlignedSegment(
            segment_id=0, time_range=TimeRange(start=0.0, end=1.0),
            original_text="hello", translated_text="hola",
        )
        assert a.translated_text == "hola"

    def test_tts_output_segment(self):
        t = TTSOutputSegment(
            segment_id=0, time_range=TimeRange(start=0.0, end=1.0),
            audio_path="/tmp/tts.wav", duration_seconds=1.0, pitch_shift_delta=0.0,
        )
        assert t.duration_seconds == pytest.approx(1.0)

    def test_duration_aligned_segment(self):
        d = DurationAlignedSegment(
            segment_id=0, time_range=TimeRange(start=0.0, end=1.0),
            audio_path="/tmp/dur.wav", adjusted_duration=0.95, stretch_ratio=1.05,
        )
        assert d.stretch_ratio == pytest.approx(1.05)

    def test_rms_processed_segment(self):
        r = RMSProcessedSegment(
            segment_id=0, time_range=TimeRange(start=0.0, end=1.0),
            audio_path="/tmp/rms.wav", rms_adjustment_delta=0.02,
        )
        assert r.rms_adjustment_delta == pytest.approx(0.02)

    def test_mixed_audio(self):
        m = MixedAudio(audio_path="/tmp/mixed.wav", integrated_lufs=-14.0, peak_db=-1.0)
        assert m.integrated_lufs == pytest.approx(-14.0)

    def test_audio_input_contract(self):
        c = AudioInputContract(
            silent_video_path="/tmp/video.mp4",
            translated_srt_path="/tmp/subs.srt",
            metadata={"fps": 30},
        )
        assert c.silent_video_path == "/tmp/video.mp4"

    def test_audio_result(self):
        r = AudioResult(audio_path="/tmp/result.wav", metadata={"duration": 10.0})
        assert r.audio_path == "/tmp/result.wav"


# ============================================================
# 2  Video DTO Construction & Validation
# ============================================================


class TestVideoDTOConstruction:

    def test_roi_box_valid(self):
        roi = ROIBox(x=10, y=20, width=100, height=50)
        assert roi.to_dict() == {"x": 10, "y": 20, "width": 100, "height": 50}

    def test_roi_box_negative_coordinates_rejected(self):
        with pytest.raises(ValueError):
            ROIBox(x=-1, y=0, width=100, height=50)

    def test_roi_box_zero_dimension_rejected(self):
        with pytest.raises(ValueError):
            ROIBox(x=0, y=0, width=0, height=50)

    def test_roi_box_roundtrip(self):
        original = ROIBox(x=5, y=10, width=200, height=100)
        assert ROIBox.from_dict(original.to_dict()) == original

    def test_mapping_entry_valid(self):
        entry = MappingEntry(source_range=(0, 50), target_hue_range=(100, 150))
        assert entry.source_range == (0, 50)

    def test_mapping_entry_invalid_range(self):
        with pytest.raises(ValueError):
            MappingEntry(source_range=(100, 50), target_hue_range=(0, 50))

    def test_mapping_config_valid(self):
        cfg = MappingConfig(
            version="1.0.0", engine_mode="HSV",
            mappings=(MappingEntry(source_range=(0, 50), target_hue_range=(100, 150)),),
        )
        assert cfg.version == "1.0.0"

    def test_mapping_config_invalid_version(self):
        with pytest.raises(ValueError):
            MappingConfig(
                version="not-semver", engine_mode="HSV",
                mappings=(MappingEntry(source_range=(0, 50), target_hue_range=(100, 150)),),
            )

    def test_mapping_config_invalid_engine_mode(self):
        with pytest.raises(ValueError):
            MappingConfig(
                version="1.0.0", engine_mode="RGB",
                mappings=(MappingEntry(source_range=(0, 50), target_hue_range=(100, 150)),),
            )

    def test_mapping_config_roundtrip(self):
        original = MappingConfig(
            version="2.0.0", engine_mode="LAB",
            mappings=(MappingEntry(source_range=(10, 20), target_hue_range=(30, 40)),),
        )
        assert MappingConfig.from_dict(original.to_dict()) == original

    def test_video_metadata_valid(self):
        meta = VideoMetadata(width=1920, height=1080, fps=30.0, frame_count=900)
        assert meta.width == 1920

    def test_video_metadata_invalid_dimensions(self):
        with pytest.raises(ValueError):
            VideoMetadata(width=0, height=1080, fps=30.0, frame_count=100)

    def test_video_result_valid(self):
        meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)
        result = VideoResult(output_path="/tmp/out.mp4", subtitle_path=None, metadata=meta)
        assert result.output_path == "/tmp/out.mp4"

    def test_video_result_empty_path_rejected(self):
        meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)
        with pytest.raises(ValueError):
            VideoResult(output_path="", subtitle_path=None, metadata=meta)

    def test_video_result_roundtrip(self):
        meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)
        original = VideoResult(output_path="/tmp/test.mp4", subtitle_path="/tmp/subs.srt", metadata=meta)
        assert VideoResult.from_dict(original.to_dict()) == original

    def test_video_output_contract(self):
        meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)
        result = VideoResult(output_path="/tmp/out.mp4", subtitle_path=None, metadata=meta)
        contract = VideoOutputContract(result=result)
        assert isinstance(contract.result, VideoResult)


# ============================================================
# 3  ConfigLoader
# ============================================================


class TestConfigLoader:
    """ConfigLoader resolution: defaults ← JSON ← CLI."""

    def test_resolve_defaults_only(self):
        resolved = ConfigLoader.resolve()
        assert resolved["engine_mode"] == "HSV"
        assert resolved["face_fusion_enabled"] == "false"

    def test_resolve_cli_overrides_defaults(self):
        resolved = ConfigLoader.resolve(cli_args={"engine_mode": "LAB"})
        assert resolved["engine_mode"] == "LAB"

    def test_resolve_json_overridden_by_cli(self):
        resolved = ConfigLoader.resolve(
            json_config={"engine_mode": "LAB"},
            cli_args={"engine_mode": "HSV"},
        )
        assert resolved["engine_mode"] == "HSV"

    def test_resolve_returns_sorted_keys(self):
        resolved = ConfigLoader.resolve()
        keys = list(resolved.keys())
        assert keys == sorted(keys)

    def test_resolved_config_immutable(self):
        cfg = ResolvedConfig(configuration_parameters={"key": "value"})
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.configuration_parameters = {}

    def test_resolved_config_from_dict(self):
        cfg = ResolvedConfig.from_dict({"a": "1", "b": "2"})
        assert cfg.values["a"] == "1"

    def test_load_json_file(self, tmp_path):
        p = tmp_path / "config.json"
        p.write_text(json.dumps({"engine_mode": "LAB"}))
        loaded = ConfigLoader.load_json(str(p))
        assert loaded["engine_mode"] == "LAB"

    def test_load_json_missing_file(self):
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load_json("/nonexistent/config.json")


# ============================================================
# 4  PipelineContext
# ============================================================


class TestPipelineContext:

    def test_factory_creates_context(self):
        cfg = ResolvedConfig.from_dict({"key": "value"})
        ctx = PipelineContextFactory.create(cfg)
        assert isinstance(ctx, PipelineContext)
        assert ctx.config.values["key"] == "value"

    def test_factory_rejects_non_resolved_config(self):
        with pytest.raises(TypeError):
            PipelineContextFactory.create({"key": "value"})


# ============================================================
# 5  WAV I/O
# ============================================================


class TestWavIO:

    def test_write_and_read_roundtrip(self, tmp_path):
        path = str(tmp_path / "test.wav")
        samples = np.sin(np.linspace(0, 2 * np.pi, 44100)).astype(np.float64)
        _write_wav(path, samples, sample_rate=44100)

        read_samples, rate = _read_wav(path)
        assert rate == 44100
        assert len(read_samples) == 44100
        np.testing.assert_allclose(read_samples, samples, atol=1e-4)


# ============================================================
# 6  Versioning & Run Metadata
# ============================================================


class TestVersioning:

    def test_generate_run_id_deterministic(self):
        config = {"a": "1", "b": "2"}
        assert generate_run_id(config) == generate_run_id(config)
        assert generate_run_id(config).startswith("run_")

    def test_different_configs_produce_different_ids(self):
        assert generate_run_id({"a": "1"}) != generate_run_id({"a": "2"})

    def test_compute_config_hash_length(self):
        assert len(compute_config_hash({"key": "value"})) == 16

    def test_compute_file_hash_missing_returns_empty(self):
        assert compute_file_hash("/nonexistent/file") == ""

    def test_create_run_metadata(self):
        meta = create_run_metadata({"a": "1"})
        assert meta.run_id.startswith("run_")
        assert meta.pipeline_version == "3.2.0"

    def test_save_run_metadata(self, tmp_path):
        meta = create_run_metadata({"a": "1"})
        path = save_run_metadata(meta, str(tmp_path))
        assert os.path.isfile(path)
        with open(path) as f:
            data = json.load(f)
        assert data["run_id"] == meta.run_id


# ============================================================
# 7  StageCheckpoint
# ============================================================


class TestStageCheckpoint:

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.checkpoint = StageCheckpoint(str(tmp_path))

    def test_save_and_load(self):
        self.checkpoint.save("stage1", {"result": "ok"})
        assert self.checkpoint.load("stage1")["result"] == "ok"

    def test_exists(self):
        assert not self.checkpoint.exists("stage1")
        self.checkpoint.save("stage1", {"x": 1})
        assert self.checkpoint.exists("stage1")

    def test_load_missing_returns_none(self):
        assert self.checkpoint.load("nonexistent") is None

    def test_clear(self):
        self.checkpoint.save("s1", {"a": 1})
        self.checkpoint.save("s2", {"b": 2})
        self.checkpoint.clear()
        assert not self.checkpoint.exists("s1")
        assert not self.checkpoint.exists("s2")


# ============================================================
# 8  CheckpointRunner
# ============================================================


class TestCheckpointRunner:

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.checkpoint = StageCheckpoint(str(tmp_path))
        self.metrics = PipelineMetrics("test-run")

    def test_fresh_execution(self):
        runner = CheckpointRunner(self.checkpoint, self.metrics)
        meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)
        result = runner.run_stage(
            "TestStage",
            lambda: VideoResult(output_path="/tmp/out.mp4", subtitle_path=None, metadata=meta),
        )
        assert isinstance(result, VideoResult)
        assert self.checkpoint.exists("TestStage")

    def test_resumes_from_checkpoint(self):
        runner = CheckpointRunner(
            self.checkpoint, self.metrics,
            type_registry={"VideoResult": VideoResult},
        )
        meta = VideoMetadata(width=320, height=240, fps=30.0, frame_count=10)
        original = VideoResult(output_path="/tmp/out.mp4", subtitle_path=None, metadata=meta)
        self.checkpoint.save("TestStage", {"__type__": "VideoResult", "data": original.to_dict()})

        call_count = 0
        def stage_fn():
            nonlocal call_count
            call_count += 1
        result = runner.run_stage("TestStage", stage_fn)
        assert call_count == 0
        assert isinstance(result, VideoResult)
        assert result.output_path == "/tmp/out.mp4"

    def test_serialization_error_on_missing_to_dict(self):
        runner = CheckpointRunner(self.checkpoint, self.metrics)
        with pytest.raises(CheckpointSerializationError):
            runner.run_stage("BadStage", lambda: "plain string")


# ============================================================
# 9  RunFolderManager
# ============================================================


class TestRunFolderManager:

    def test_create_and_paths(self, tmp_path):
        mgr = RunFolderManager(base_dir=str(tmp_path))
        run_path = mgr.create("test_run")
        assert os.path.isdir(run_path)
        assert os.path.isdir(mgr.artifacts_dir)
        assert os.path.isdir(mgr.checkpoints_dir)
        assert os.path.isdir(mgr.logs_dir)
        assert "test_run" in mgr.artifact_path("test.mp4")
        assert "checkpoints" in mgr.checkpoint_path()
        assert "pipeline.log" in mgr.log_path()
        assert "debug_report.json" in mgr.debug_report_path()


# ============================================================
# 10  Metrics Instrumentation
# ============================================================


class TestMetricsInstrumentation:

    def test_records_success(self):
        metrics = PipelineMetrics("test-run")

        @instrument_stage(metrics, "MyStage")
        def my_fn():
            return 42

        assert my_fn() == 42
        summary = metrics.get_summary()
        assert summary["total_stages"] == 1
        assert summary["passed"] == 1

    def test_records_failure(self):
        metrics = PipelineMetrics("test-run")

        @instrument_stage(metrics, "FailStage")
        def bad_fn():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            bad_fn()
        assert metrics.get_summary()["failed"] == 1


# ============================================================
# 11  stage_timer Context Manager
# ============================================================


class TestStageTimer:

    def test_records_elapsed(self):
        with stage_timer("TestStage") as m:
            _ = sum(range(100))
        assert "elapsed_ms" in m
        assert m["elapsed_ms"] >= 0.0
        assert m["stage"] == "TestStage"
